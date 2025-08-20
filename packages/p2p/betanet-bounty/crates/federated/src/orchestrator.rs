//! Federated Learning Round Orchestrator
//!
//! Coordinates FL rounds via MLS groups and manages participant cohorts.
//! Publishes RoundPlans, tracks participation, and handles failures gracefully.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use bytes::Bytes;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc};
use tokio::time::{interval, timeout, Instant};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use agent_fabric::{
    groups::{GroupMessageType, TrainingAction},
    AgentFabric, AgentId, AgentMessage, AgentResponse, GroupConfig, GroupMessage,
};
use twin_vault::{Receipt, TwinId, TwinManager, TwinOperation};

use crate::{
    AggregationResult, AggregationStats, DeviceCapabilities, DeviceType, FLSession, FederatedError,
    ModelParameters, ParticipantId, Result, RoundId, SessionStatus, TrainingConfig, TrainingResult,
};

/// Round orchestrator that coordinates federated learning via MLS groups
pub struct RoundOrchestrator {
    /// Agent fabric for communication
    agent_fabric: Arc<AgentFabric>,
    /// Twin manager for state synchronization
    twin_manager: Arc<TwinManager>,
    /// Active sessions
    sessions: Arc<RwLock<HashMap<String, FLSession>>>,
    /// Active cohorts per session
    cohorts: Arc<RwLock<HashMap<String, CohortManager>>>,
    /// Round plans publisher
    round_plans: Arc<RwLock<HashMap<RoundId, RoundPlan>>>,
    /// Event broadcaster for coordination
    event_sender: broadcast::Sender<OrchestrationEvent>,
    /// Command receiver for external control
    command_receiver: Arc<tokio::sync::Mutex<mpsc::Receiver<OrchestrationCommand>>>,
    /// Orchestrator configuration
    config: OrchestratorConfig,
}

impl RoundOrchestrator {
    /// Create new round orchestrator
    pub async fn new(
        agent_fabric: Arc<AgentFabric>,
        twin_manager: Arc<TwinManager>,
        config: OrchestratorConfig,
    ) -> Result<(Self, mpsc::Sender<OrchestrationCommand>)> {
        let (event_sender, _) = broadcast::channel(1000);
        let (command_sender, command_receiver) = mpsc::channel(100);

        let orchestrator = Self {
            agent_fabric,
            twin_manager,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            cohorts: Arc::new(RwLock::new(HashMap::new())),
            round_plans: Arc::new(RwLock::new(HashMap::new())),
            event_sender,
            command_receiver: Arc::new(tokio::sync::Mutex::new(command_receiver)),
            config,
        };

        Ok((orchestrator, command_sender))
    }

    /// Start the orchestrator event loop
    pub async fn start(&self) -> Result<()> {
        info!("Starting federated learning orchestrator");

        // Start background tasks
        let heartbeat_task = self.start_heartbeat_task();
        let coordination_task = self.start_coordination_task();
        let command_handler_task = self.start_command_handler();

        // Join all tasks
        tokio::select! {
            result = heartbeat_task => {
                error!("Heartbeat task ended: {:?}", result);
                result?
            }
            result = coordination_task => {
                error!("Coordination task ended: {:?}", result);
                result?
            }
            result = command_handler_task => {
                error!("Command handler ended: {:?}", result);
                result?
            }
        }

        Ok(())
    }

    /// Create a new federated learning session
    pub async fn create_session(&self, config: TrainingConfig) -> Result<String> {
        let session_id = Uuid::new_v4().to_string();
        let session = FLSession::new(session_id.clone(), config);

        info!("Creating FL session: {}", session_id);

        // Create MLS group for this session
        let group_config = GroupConfig {
            group_id: format!("fl-session-{}", session_id),
            max_members: session.target_participants as usize,
            admin_only_add: true,
            admin_only_remove: true,
            require_unanimous_votes: false,
            vote_timeout_seconds: self.config.vote_timeout_sec,
            ..Default::default()
        };

        self.agent_fabric
            .join_group(group_config.group_id.clone(), group_config)
            .await
            .map_err(FederatedError::AgentFabricError)?;

        // Create cohort manager
        let cohort_manager = CohortManager::new(
            session_id.clone(),
            session.min_participants,
            session.target_participants,
        );

        // Store session and cohort
        self.sessions.write().insert(session_id.clone(), session);
        self.cohorts
            .write()
            .insert(session_id.clone(), cohort_manager);

        // Publish session creation event
        let event = OrchestrationEvent::SessionCreated {
            session_id: session_id.clone(),
            group_id: format!("fl-session-{}", session_id),
        };
        let _ = self.event_sender.send(event);

        Ok(session_id)
    }

    /// Add participant to session
    pub async fn add_participant(
        &self,
        session_id: &str,
        participant: ParticipantId,
    ) -> Result<()> {
        info!(
            "Adding participant {} to session {}",
            participant.agent_id, session_id
        );

        // Add to MLS group
        let group_id = format!("fl-session-{}", session_id);
        // Note: In real MLS implementation, would add member to group
        // For now, we track in cohort manager

        // Add to cohort
        if let Some(cohort) = self.cohorts.write().get_mut(session_id) {
            cohort.add_participant(participant.clone())?;

            // Check if we can start the session
            if cohort.can_start_round()
                && self.get_session_status(session_id)? == SessionStatus::WaitingForParticipants
            {
                self.update_session_status(session_id, SessionStatus::InProgress)
                    .await?;
                self.start_first_round(session_id).await?;
            }

            let event = OrchestrationEvent::ParticipantAdded {
                session_id: session_id.to_string(),
                participant_id: participant,
            };
            let _ = self.event_sender.send(event);

            Ok(())
        } else {
            Err(FederatedError::SessionNotFound(session_id.to_string()))
        }
    }

    /// Start a new federated learning round
    pub async fn start_round(&self, session_id: &str) -> Result<RoundId> {
        info!("Starting new FL round for session: {}", session_id);

        // Get session and cohort
        let session = self.get_session(session_id)?;
        let mut cohorts = self.cohorts.write();
        let cohort = cohorts
            .get_mut(session_id)
            .ok_or_else(|| FederatedError::SessionNotFound(session_id.to_string()))?;

        // Check if we can start a round
        if !cohort.can_start_round() {
            return Err(FederatedError::InsufficientParticipants {
                got: cohort.active_participants(),
                need: session.min_participants,
            });
        }

        // Create round ID
        let round_id = RoundId::new(
            session_id.to_string(),
            cohort.current_round + 1,
            session.created_at,
        );

        // Create round plan
        let round_plan = RoundPlan::new(
            round_id.clone(),
            cohort.get_active_participants(),
            session.config.clone(),
            Duration::from_secs(session.round_timeout_sec),
        );

        // Update cohort state
        cohort.start_round(round_id.clone());

        // Store round plan
        self.round_plans
            .write()
            .insert(round_id.clone(), round_plan.clone());

        // Publish round plan via MLS group
        self.publish_round_plan(&round_plan).await?;

        // Publish event
        let event = OrchestrationEvent::RoundStarted {
            round_id: round_id.clone(),
            participants: cohort.get_active_participants(),
        };
        let _ = self.event_sender.send(event);

        info!("Started FL round: {}", round_id);
        Ok(round_id)
    }

    /// Collect training result from participant
    pub async fn collect_training_result(
        &self,
        round_id: &RoundId,
        result: TrainingResult,
    ) -> Result<()> {
        info!(
            "Collecting training result from {} for round {}",
            result.participant_id.agent_id, round_id
        );

        // Verify the result is for the correct round
        if result.round_id != *round_id {
            return Err(FederatedError::ModelError(
                "Training result round ID mismatch".to_string(),
            ));
        }

        // Store result via twin vault for persistence
        let twin_id = TwinId::new(
            result.participant_id.agent_id.clone(),
            format!("fl-result-{}", round_id),
        );

        let result_data = bincode::serialize(&result)
            .map_err(|e| FederatedError::SerializationError(e.to_string()))?;

        let operation = TwinOperation::Write {
            key: format!("training-result-{}", round_id),
            value: Bytes::from(result_data),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        };

        self.twin_manager
            .perform_operation(twin_id, operation, result.participant_id.agent_id.clone())
            .await
            .map_err(FederatedError::TwinVaultError)?;

        // Update cohort with result
        if let Some(cohort) = self.cohorts.write().get_mut(&round_id.session_id) {
            cohort.add_training_result(result.clone());

            // Check if round is complete
            if cohort.is_round_complete() {
                self.complete_round(round_id).await?;
            }
        }

        let event = OrchestrationEvent::TrainingResultReceived {
            round_id: round_id.clone(),
            participant_id: result.participant_id,
        };
        let _ = self.event_sender.send(event);

        Ok(())
    }

    /// Get orchestration events
    pub fn subscribe_events(&self) -> broadcast::Receiver<OrchestrationEvent> {
        self.event_sender.subscribe()
    }

    // Private methods

    async fn start_heartbeat_task(&self) -> Result<()> {
        let mut interval = interval(Duration::from_secs(self.config.heartbeat_interval_sec));

        loop {
            interval.tick().await;

            // Check for timed out rounds
            self.check_round_timeouts().await?;

            // Update participant health
            self.update_participant_health().await?;

            debug!("Orchestrator heartbeat");
        }
    }

    async fn start_coordination_task(&self) -> Result<()> {
        let mut interval = interval(Duration::from_secs(self.config.coordination_interval_sec));

        loop {
            interval.tick().await;

            // Check for sessions that need round progression
            self.check_round_progression().await?;

            debug!("Coordination check");
        }
    }

    async fn start_command_handler(&self) -> Result<()> {
        let mut receiver = self.command_receiver.lock().await;

        while let Some(command) = receiver.recv().await {
            match command {
                OrchestrationCommand::StartSession { session_id } => {
                    if let Err(e) = self
                        .update_session_status(&session_id, SessionStatus::InProgress)
                        .await
                    {
                        error!("Failed to start session {}: {}", session_id, e);
                    }
                }
                OrchestrationCommand::StopSession { session_id } => {
                    if let Err(e) = self
                        .update_session_status(&session_id, SessionStatus::Cancelled)
                        .await
                    {
                        error!("Failed to stop session {}: {}", session_id, e);
                    }
                }
                OrchestrationCommand::ForceRound { session_id } => {
                    if let Err(e) = self.start_round(&session_id).await {
                        error!("Failed to force round for session {}: {}", session_id, e);
                    }
                }
            }
        }

        Ok(())
    }

    async fn publish_round_plan(&self, round_plan: &RoundPlan) -> Result<()> {
        let group_id = format!("fl-session-{}", round_plan.round_id.session_id);

        let plan_data = bincode::serialize(round_plan)
            .map_err(|e| FederatedError::SerializationError(e.to_string()))?;

        let message = GroupMessage {
            message_id: Uuid::new_v4().to_string(),
            from: self.agent_fabric.node_id().clone(),
            message_type: GroupMessageType::Training {
                session_id: round_plan.round_id.session_id.clone(),
                action: TrainingAction::StartSession,
            },
            payload: Bytes::from(plan_data),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        self.agent_fabric
            .send_to_group(group_id, message)
            .await
            .map_err(FederatedError::AgentFabricError)?;

        info!("Published round plan for round: {}", round_plan.round_id);
        Ok(())
    }

    async fn complete_round(&self, round_id: &RoundId) -> Result<()> {
        info!("Completing FL round: {}", round_id);

        // Aggregate results
        let aggregation_result = self.aggregate_round_results(round_id).await?;

        // Update session with results
        if let Some(session) = self.sessions.write().get_mut(&round_id.session_id) {
            // Check for convergence
            if aggregation_result.stats.converged || round_id.round_number >= session.max_rounds {
                session.status = if aggregation_result.stats.converged {
                    SessionStatus::Converged
                } else {
                    SessionStatus::MaxRoundsReached
                };

                info!(
                    "Session {} completed with status: {:?}",
                    round_id.session_id, session.status
                );
            }
        }

        // Publish aggregation result
        self.publish_aggregation_result(&aggregation_result).await?;

        let event = OrchestrationEvent::RoundCompleted {
            round_id: round_id.clone(),
            result: aggregation_result,
        };
        let _ = self.event_sender.send(event);

        Ok(())
    }

    async fn aggregate_round_results(&self, round_id: &RoundId) -> Result<AggregationResult> {
        // Get training results from cohort
        let cohorts = self.cohorts.read();
        let cohort = cohorts
            .get(&round_id.session_id)
            .ok_or_else(|| FederatedError::SessionNotFound(round_id.session_id.clone()))?;

        let results = cohort.get_training_results();
        if results.is_empty() {
            return Err(FederatedError::AggregationError(
                "No training results to aggregate".to_string(),
            ));
        }

        // Simple FedAvg aggregation (will be enhanced in fedavg_secureagg.rs)
        let mut total_examples = 0u64;
        let mut weighted_loss = 0.0f32;
        let mut weighted_accuracy = 0.0f32;
        let participants: Vec<ParticipantId> =
            results.iter().map(|r| r.participant_id.clone()).collect();

        for result in &results {
            total_examples += result.metrics.num_examples as u64;
            weighted_loss += result.metrics.training_loss * result.metrics.num_examples as f32;
            weighted_accuracy +=
                result.metrics.training_accuracy * result.metrics.num_examples as f32;
        }

        let avg_loss = weighted_loss / total_examples as f32;
        let avg_accuracy = weighted_accuracy / total_examples as f32;

        // Create aggregated model (placeholder - actual aggregation in fedavg_secureagg.rs)
        let global_model = ModelParameters::new(
            format!("global-{}", round_id),
            Bytes::from("aggregated_weights_placeholder"),
            results[0].model_update.metadata.clone(),
        );

        let stats = AggregationStats {
            num_participants: results.len() as u32,
            total_examples,
            avg_training_loss: avg_loss,
            avg_training_accuracy: avg_accuracy,
            improvement: 0.0,          // Calculate based on previous round
            converged: avg_loss < 0.1, // Simple convergence criterion
        };

        Ok(AggregationResult {
            round_id: round_id.clone(),
            global_model,
            stats,
            participants,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        })
    }

    async fn publish_aggregation_result(&self, result: &AggregationResult) -> Result<()> {
        let group_id = format!("fl-session-{}", result.round_id.session_id);

        let result_data = bincode::serialize(result)
            .map_err(|e| FederatedError::SerializationError(e.to_string()))?;

        let message = GroupMessage {
            message_id: Uuid::new_v4().to_string(),
            from: self.agent_fabric.node_id().clone(),
            message_type: GroupMessageType::Training {
                session_id: result.round_id.session_id.clone(),
                action: TrainingAction::UpdateModel,
            },
            payload: Bytes::from(result_data),
            timestamp: result.timestamp,
        };

        self.agent_fabric
            .send_to_group(group_id, message)
            .await
            .map_err(FederatedError::AgentFabricError)?;

        info!(
            "Published aggregation result for round: {}",
            result.round_id
        );
        Ok(())
    }

    async fn check_round_timeouts(&self) -> Result<()> {
        let now = Instant::now();
        let mut rounds_to_timeout = Vec::new();

        // Check for timed out rounds
        {
            let round_plans = self.round_plans.read();
            for (round_id, plan) in round_plans.iter() {
                if now.duration_since(plan.start_time) > plan.timeout {
                    rounds_to_timeout.push(round_id.clone());
                }
            }
        }

        // Handle timeouts
        for round_id in rounds_to_timeout {
            warn!("Round timeout: {}", round_id);
            self.handle_round_timeout(&round_id).await?;
        }

        Ok(())
    }

    async fn handle_round_timeout(&self, round_id: &RoundId) -> Result<()> {
        // Force completion with available results
        if let Some(cohort) = self.cohorts.read().get(&round_id.session_id) {
            if !cohort.get_training_results().is_empty() {
                info!(
                    "Completing round {} with {} results after timeout",
                    round_id,
                    cohort.get_training_results().len()
                );
                self.complete_round(round_id).await?;
            } else {
                warn!("Round {} timed out with no results, cancelling", round_id);
                // Handle failed round
            }
        }

        let event = OrchestrationEvent::RoundTimeout {
            round_id: round_id.clone(),
        };
        let _ = self.event_sender.send(event);

        Ok(())
    }

    async fn update_participant_health(&self) -> Result<()> {
        // Check participant connectivity and update cohort status
        // This would integrate with actual health checking mechanisms
        Ok(())
    }

    async fn check_round_progression(&self) -> Result<()> {
        // Check if any sessions need to start new rounds
        let sessions = self.sessions.read().clone();

        for (session_id, session) in &sessions {
            if session.status == SessionStatus::InProgress {
                if let Some(cohort) = self.cohorts.read().get(session_id) {
                    if cohort.can_start_round() && cohort.current_round_result.is_some() {
                        // Start next round
                        if let Err(e) = self.start_round(&session_id).await {
                            error!(
                                "Failed to start next round for session {}: {}",
                                session_id, e
                            );
                        }
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    async fn start_first_round(&self, session_id: &str) -> Result<()> {
        info!("Starting first round for session: {}", session_id);
        self.start_round(session_id).await?;
        Ok(())
    }

    async fn update_session_status(&self, session_id: &str, status: SessionStatus) -> Result<()> {
        if let Some(session) = self.sessions.write().get_mut(session_id) {
            session.status = status;
            info!(
                "Updated session {} status to {:?}",
                session_id, session.status
            );
            Ok(())
        } else {
            Err(FederatedError::SessionNotFound(session_id.to_string()))
        }
    }

    fn get_session(&self, session_id: &str) -> Result<FLSession> {
        self.sessions
            .read()
            .get(session_id)
            .cloned()
            .ok_or_else(|| FederatedError::SessionNotFound(session_id.to_string()))
    }

    fn get_session_status(&self, session_id: &str) -> Result<SessionStatus> {
        Ok(self.get_session(session_id)?.status)
    }
}

/// Round plan published to participants via MLS groups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundPlan {
    /// Round identifier
    pub round_id: RoundId,
    /// Selected participants for this round
    pub participants: Vec<ParticipantId>,
    /// Training configuration
    pub training_config: TrainingConfig,
    /// Round timeout
    pub timeout: Duration,
    /// Global model for this round
    pub global_model: Option<ModelParameters>,
    /// Round start time
    #[serde(skip, default = "tokio::time::Instant::now")]
    pub start_time: Instant,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl RoundPlan {
    pub fn new(
        round_id: RoundId,
        participants: Vec<ParticipantId>,
        training_config: TrainingConfig,
        timeout: Duration,
    ) -> Self {
        Self {
            round_id,
            participants,
            training_config,
            timeout,
            global_model: None,
            start_time: Instant::now(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_global_model(mut self, model: ModelParameters) -> Self {
        self.global_model = Some(model);
        self
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Manages participant cohorts for federated learning sessions
#[derive(Debug, Clone)]
pub struct CohortManager {
    /// Session identifier
    pub session_id: String,
    /// Minimum participants to start round
    pub min_participants: u32,
    /// Target participants per round
    pub target_participants: u32,
    /// Current round number
    pub current_round: u64,
    /// All registered participants
    pub participants: HashMap<AgentId, ParticipantId>,
    /// Active participants (currently online and healthy)
    pub active_participants: HashSet<AgentId>,
    /// Participants selected for current round
    pub round_participants: HashSet<AgentId>,
    /// Training results for current round
    pub training_results: HashMap<AgentId, TrainingResult>,
    /// Current round aggregation result
    pub current_round_result: Option<AggregationResult>,
    /// Participant health status
    pub participant_health: HashMap<AgentId, ParticipantHealth>,
}

impl CohortManager {
    pub fn new(session_id: String, min_participants: u32, target_participants: u32) -> Self {
        Self {
            session_id,
            min_participants,
            target_participants,
            current_round: 0,
            participants: HashMap::new(),
            active_participants: HashSet::new(),
            round_participants: HashSet::new(),
            training_results: HashMap::new(),
            current_round_result: None,
            participant_health: HashMap::new(),
        }
    }

    pub fn add_participant(&mut self, participant: ParticipantId) -> Result<()> {
        let agent_id = participant.agent_id.clone();

        if self.participants.contains_key(&agent_id) {
            return Ok(()); // Already added
        }

        self.participants.insert(agent_id.clone(), participant);
        self.active_participants.insert(agent_id.clone());
        self.participant_health
            .insert(agent_id.clone(), ParticipantHealth::new());

        info!(
            "Added participant {} to cohort for session {}",
            agent_id, self.session_id
        );
        Ok(())
    }

    pub fn remove_participant(&mut self, agent_id: &AgentId) {
        self.participants.remove(agent_id);
        self.active_participants.remove(agent_id);
        self.round_participants.remove(agent_id);
        self.training_results.remove(agent_id);
        self.participant_health.remove(agent_id);

        info!(
            "Removed participant {} from cohort for session {}",
            agent_id, self.session_id
        );
    }

    pub fn can_start_round(&self) -> bool {
        self.active_participants.len() >= self.min_participants as usize
    }

    pub fn active_participants(&self) -> u32 {
        self.active_participants.len() as u32
    }

    pub fn get_active_participants(&self) -> Vec<ParticipantId> {
        self.active_participants
            .iter()
            .filter_map(|id| self.participants.get(id).cloned())
            .collect()
    }

    pub fn start_round(&mut self, round_id: RoundId) {
        self.current_round = round_id.round_number;
        self.training_results.clear();
        self.current_round_result = None;

        // Select participants for this round
        self.round_participants = self.select_round_participants();

        info!(
            "Started round {} with {} participants",
            self.current_round,
            self.round_participants.len()
        );
    }

    pub fn add_training_result(&mut self, result: TrainingResult) {
        let agent_id = result.participant_id.agent_id.clone();

        if !self.round_participants.contains(&agent_id) {
            warn!("Received result from non-participating agent: {}", agent_id);
            return;
        }

        self.training_results.insert(agent_id.clone(), result);
        info!(
            "Received training result from {} ({}/{})",
            agent_id,
            self.training_results.len(),
            self.round_participants.len()
        );
    }

    pub fn is_round_complete(&self) -> bool {
        // Round is complete when we have results from enough participants
        let min_results = (self.round_participants.len() as f32 * 0.7) as usize; // 70% threshold
        self.training_results.len() >= min_results.max(1)
    }

    pub fn get_training_results(&self) -> Vec<TrainingResult> {
        self.training_results.values().cloned().collect()
    }

    pub fn update_participant_health(&mut self, agent_id: &AgentId, is_healthy: bool) {
        if let Some(health) = self.participant_health.get_mut(agent_id) {
            health.update(is_healthy);

            // Update active status based on health
            if health.is_healthy() {
                self.active_participants.insert(agent_id.clone());
            } else {
                self.active_participants.remove(agent_id);
                warn!("Participant {} marked as unhealthy", agent_id);
            }
        }
    }

    fn select_round_participants(&self) -> HashSet<AgentId> {
        // Simple selection: use all active participants up to target
        let target_size = self
            .target_participants
            .min(self.active_participants.len() as u32) as usize;
        self.active_participants
            .iter()
            .take(target_size)
            .cloned()
            .collect()
    }
}

/// Participant health tracking
#[derive(Debug, Clone)]
pub struct ParticipantHealth {
    pub last_seen: SystemTime,
    pub consecutive_failures: u32,
    pub success_rate: f32,
    pub total_attempts: u32,
    pub successful_attempts: u32,
}

impl ParticipantHealth {
    pub fn new() -> Self {
        Self {
            last_seen: SystemTime::now(),
            consecutive_failures: 0,
            success_rate: 1.0,
            total_attempts: 0,
            successful_attempts: 0,
        }
    }

    pub fn update(&mut self, success: bool) {
        self.last_seen = SystemTime::now();
        self.total_attempts += 1;

        if success {
            self.consecutive_failures = 0;
            self.successful_attempts += 1;
        } else {
            self.consecutive_failures += 1;
        }

        self.success_rate = self.successful_attempts as f32 / self.total_attempts as f32;
    }

    pub fn is_healthy(&self) -> bool {
        self.consecutive_failures < 3 && self.success_rate > 0.5
    }

    pub fn is_recently_active(&self, threshold: Duration) -> bool {
        self.last_seen.elapsed().unwrap_or(Duration::MAX) < threshold
    }
}

/// Orchestration events published by the orchestrator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrchestrationEvent {
    SessionCreated {
        session_id: String,
        group_id: String,
    },
    ParticipantAdded {
        session_id: String,
        participant_id: ParticipantId,
    },
    RoundStarted {
        round_id: RoundId,
        participants: Vec<ParticipantId>,
    },
    TrainingResultReceived {
        round_id: RoundId,
        participant_id: ParticipantId,
    },
    RoundCompleted {
        round_id: RoundId,
        result: AggregationResult,
    },
    RoundTimeout {
        round_id: RoundId,
    },
    SessionCompleted {
        session_id: String,
        final_result: AggregationResult,
    },
}

/// Commands for controlling the orchestrator
#[derive(Debug, Clone)]
pub enum OrchestrationCommand {
    StartSession { session_id: String },
    StopSession { session_id: String },
    ForceRound { session_id: String },
}

/// Orchestrator configuration
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Heartbeat interval in seconds
    pub heartbeat_interval_sec: u64,
    /// Coordination check interval in seconds
    pub coordination_interval_sec: u64,
    /// MLS group vote timeout in seconds
    pub vote_timeout_sec: u64,
    /// Maximum concurrent sessions
    pub max_concurrent_sessions: u32,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval_sec: 30,
            coordination_interval_sec: 60,
            vote_timeout_sec: 300,
            max_concurrent_sessions: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_fabric::AgentFabric;
    use tempfile::TempDir;
    use twin_vault::{ReceiptSigner, ReceiptVerifier};

    #[tokio::test]
    async fn test_round_plan_creation() {
        let round_id = RoundId::new("test-session".to_string(), 1, 1234567890);
        let participants = vec![ParticipantId::new(
            AgentId::new("phone-001", "mobile"),
            DeviceType::Phone,
            DeviceCapabilities::default(),
        )];
        let config = TrainingConfig::default();

        let plan = RoundPlan::new(
            round_id.clone(),
            participants.clone(),
            config,
            Duration::from_secs(300),
        );

        assert_eq!(plan.round_id, round_id);
        assert_eq!(plan.participants.len(), 1);
        assert_eq!(plan.timeout, Duration::from_secs(300));
    }

    #[tokio::test]
    async fn test_cohort_manager() {
        let mut cohort = CohortManager::new("test-session".to_string(), 2, 5);

        let participant1 = ParticipantId::new(
            AgentId::new("phone-001", "mobile"),
            DeviceType::Phone,
            DeviceCapabilities::default(),
        );

        let participant2 = ParticipantId::new(
            AgentId::new("phone-002", "mobile"),
            DeviceType::Phone,
            DeviceCapabilities::default(),
        );

        // Add participants
        cohort.add_participant(participant1.clone()).unwrap();
        assert!(!cohort.can_start_round()); // Need min 2 participants

        cohort.add_participant(participant2.clone()).unwrap();
        assert!(cohort.can_start_round()); // Now have 2 participants

        // Start round
        let round_id = RoundId::new("test-session".to_string(), 1, 1234567890);
        cohort.start_round(round_id);

        assert_eq!(cohort.current_round, 1);
        assert_eq!(cohort.round_participants.len(), 2);
        assert!(!cohort.is_round_complete()); // No results yet
    }

    #[test]
    fn test_participant_health() {
        let mut health = ParticipantHealth::new();
        assert!(health.is_healthy());

        // Simulate failures
        health.update(false);
        health.update(false);
        assert!(health.is_healthy()); // Still healthy after 2 failures

        health.update(false);
        assert!(!health.is_healthy()); // Unhealthy after 3 consecutive failures

        // Recovery
        health.update(true);
        assert!(health.is_healthy()); // Healthy again after success
    }
}
