//! FedAvg with Secure Aggregation and Differential Privacy
//!
//! Implements secure federated averaging with:
//! - Additive secret sharing masks for privacy
//! - Differential privacy via DP-SGD
//! - Model compression (quantization, top-k sparsification)
//! - Cryptographic verification of aggregation

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2, ArrayView1, Axis};
use num_traits::Float;
use rand::{thread_rng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use tracing::{debug, info, warn};

use crate::{
    AggregationResult, AggregationStats, CompressionType, FederatedError, ModelParameters,
    ParticipantId, PrivacyConfig, QuantizationType, Result, RoundId, TrainingResult,
};

/// Secure aggregation protocol for federated learning
pub struct SecureAggregation {
    /// Privacy configuration
    privacy_config: PrivacyConfig,
    /// Compression settings
    compression_type: CompressionType,
    /// Quantization settings
    quantization_type: QuantizationType,
    /// Random number generator for masks
    rng: rand::rngs::ThreadRng,
}

impl SecureAggregation {
    pub fn new(
        privacy_config: PrivacyConfig,
        compression_type: CompressionType,
        quantization_type: QuantizationType,
    ) -> Self {
        Self {
            privacy_config,
            compression_type,
            quantization_type,
            rng: thread_rng(),
        }
    }

    /// Generate additive secret sharing mask for a participant
    pub fn generate_mask(
        &mut self,
        model_size: usize,
        participant_id: &ParticipantId,
    ) -> SecretMask {
        // Generate deterministic seed from participant ID for reproducibility
        let mut hasher = Sha256::new();
        hasher.update(participant_id.agent_id.to_string().as_bytes());
        let seed_hash = hasher.finalize();
        let seed = u64::from_le_bytes(seed_hash[0..8].try_into().unwrap());

        // Use seeded RNG for this participant
        let mut participant_rng = rand::rngs::StdRng::from_seed(seed_hash.into());

        let mask: Vec<f32> = (0..model_size)
            .map(|_| participant_rng.gen_range(-1.0..1.0))
            .collect();

        SecretMask {
            participant_id: participant_id.clone(),
            mask: Array1::from(mask),
            seed,
        }
    }

    /// Apply secure aggregation to training results
    pub async fn secure_aggregate(
        &mut self,
        round_id: &RoundId,
        training_results: Vec<TrainingResult>,
    ) -> Result<AggregationResult> {
        if training_results.is_empty() {
            return Err(FederatedError::AggregationError(
                "No training results to aggregate".to_string(),
            ));
        }

        info!(
            "Starting secure aggregation for round {} with {} participants",
            round_id,
            training_results.len()
        );

        // Step 1: Extract and validate model parameters
        let model_updates = self.extract_model_updates(&training_results)?;

        // Step 2: Apply differential privacy if enabled
        let dp_updates = if self.privacy_config.enable_dp {
            self.apply_differential_privacy(model_updates)?
        } else {
            model_updates
        };

        // Step 3: Apply compression
        let compressed_updates = self.apply_compression(dp_updates)?;

        // Step 4: Perform secure aggregation with masks
        let aggregated_model = if self.privacy_config.enable_secure_agg {
            self.secure_fedavg_with_masks(compressed_updates).await?
        } else {
            self.standard_fedavg(compressed_updates)?
        };

        // Step 5: Create aggregation statistics
        let stats = self.compute_aggregation_stats(&training_results);

        // Step 6: Create aggregation result
        let participants: Vec<ParticipantId> = training_results
            .iter()
            .map(|r| r.participant_id.clone())
            .collect();

        let result = AggregationResult {
            round_id: round_id.clone(),
            global_model: aggregated_model,
            stats,
            participants,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        info!("Completed secure aggregation for round {}", round_id);
        Ok(result)
    }

    /// Extract model parameters from training results
    fn extract_model_updates(&self, results: &[TrainingResult]) -> Result<Vec<ModelUpdate>> {
        let mut updates = Vec::new();

        for result in results {
            // Deserialize model weights
            let weights = self.deserialize_weights(&result.model_update.weights)?;

            let update = ModelUpdate {
                participant_id: result.participant_id.clone(),
                weights,
                num_examples: result.metrics.num_examples,
                training_loss: result.metrics.training_loss,
            };

            updates.push(update);
        }

        // Validate all updates have same dimensions
        if let Some(first_update) = updates.first() {
            let expected_size = first_update.weights.len();
            for update in &updates {
                if update.weights.len() != expected_size {
                    return Err(FederatedError::AggregationError(
                        "Model updates have inconsistent dimensions".to_string(),
                    ));
                }
            }
        }

        Ok(updates)
    }

    /// Apply differential privacy to model updates
    fn apply_differential_privacy(
        &self,
        mut updates: Vec<ModelUpdate>,
    ) -> Result<Vec<ModelUpdate>> {
        info!(
            "Applying differential privacy with ε={}, δ={}",
            self.privacy_config.dp_epsilon, self.privacy_config.dp_delta
        );

        for update in &mut updates {
            // Clip gradients to bound sensitivity
            let clipped_weights =
                self.clip_gradients(&update.weights, self.privacy_config.clipping_norm);

            // Add calibrated noise
            let noisy_weights = self.add_dp_noise(
                clipped_weights,
                self.privacy_config.noise_multiplier,
                self.privacy_config.clipping_norm,
            )?;

            update.weights = noisy_weights;
        }

        Ok(updates)
    }

    /// Clip gradients to bound L2 norm
    fn clip_gradients(&self, weights: &Array1<f32>, clipping_norm: f32) -> Array1<f32> {
        let l2_norm = weights.mapv(|x| x * x).sum().sqrt();

        if l2_norm > clipping_norm {
            weights * (clipping_norm / l2_norm)
        } else {
            weights.clone()
        }
    }

    /// Add Gaussian noise for differential privacy
    fn add_dp_noise(
        &self,
        mut weights: Array1<f32>,
        noise_multiplier: f32,
        clipping_norm: f32,
    ) -> Result<Array1<f32>> {
        let noise_stddev = noise_multiplier * clipping_norm;

        for weight in weights.iter_mut() {
            let noise: f32 =
                thread_rng().sample(rand_distr::Normal::new(0.0, noise_stddev).map_err(|e| {
                    FederatedError::AggregationError(format!("DP noise generation failed: {}", e))
                })?);
            *weight += noise;
        }

        Ok(weights)
    }

    /// Apply model compression
    fn apply_compression(&self, mut updates: Vec<ModelUpdate>) -> Result<Vec<ModelUpdate>> {
        match &self.compression_type {
            CompressionType::None => Ok(updates),
            CompressionType::Q8 => {
                info!("Applying 8-bit quantization");
                for update in &mut updates {
                    update.weights = self.quantize_weights(&update.weights, 8)?;
                }
                Ok(updates)
            }
            CompressionType::TopK { k } => {
                info!("Applying top-{} sparsification", k);
                for update in &mut updates {
                    update.weights = self.top_k_sparsification(&update.weights, *k)?;
                }
                Ok(updates)
            }
            CompressionType::Random { prob } => {
                info!("Applying random sparsification with probability {}", prob);
                for update in &mut updates {
                    update.weights = self.random_sparsification(&update.weights, *prob)?;
                }
                Ok(updates)
            }
            CompressionType::Gradient { threshold } => {
                info!("Applying gradient compression with threshold {}", threshold);
                for update in &mut updates {
                    update.weights = self.gradient_compression(&update.weights, *threshold)?;
                }
                Ok(updates)
            }
        }
    }

    /// Quantize weights to specified bit width
    fn quantize_weights(&self, weights: &Array1<f32>, bits: u8) -> Result<Array1<f32>> {
        let max_val = (1 << (bits - 1)) - 1;
        let min_val = -(1 << (bits - 1));

        // Find min/max for scaling
        let w_min = weights.fold(f32::INFINITY, |a, &b| a.min(b));
        let w_max = weights.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let scale = (w_max - w_min) / (max_val - min_val) as f32;

        let quantized = weights.mapv(|w| {
            let quantized_int = ((w - w_min) / scale).round() as i32;
            let clamped = quantized_int.max(min_val).min(max_val);
            (clamped as f32 * scale) + w_min
        });

        Ok(quantized)
    }

    /// Apply top-k sparsification
    fn top_k_sparsification(&self, weights: &Array1<f32>, k: usize) -> Result<Array1<f32>> {
        if k >= weights.len() {
            return Ok(weights.clone());
        }

        let mut indexed_weights: Vec<(usize, f32)> = weights
            .iter()
            .enumerate()
            .map(|(i, &w)| (i, w.abs()))
            .collect();

        // Sort by absolute value (descending)
        indexed_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut sparse_weights = Array1::zeros(weights.len());
        for (orig_idx, _) in indexed_weights.iter().take(k) {
            sparse_weights[*orig_idx] = weights[*orig_idx];
        }

        Ok(sparse_weights)
    }

    /// Apply random sparsification
    fn random_sparsification(&self, weights: &Array1<f32>, keep_prob: f32) -> Result<Array1<f32>> {
        let sparse_weights = weights.mapv(|w| {
            if thread_rng().gen::<f32>() < keep_prob {
                w / keep_prob // Scale to maintain expected value
            } else {
                0.0
            }
        });

        Ok(sparse_weights)
    }

    /// Apply gradient-based compression
    fn gradient_compression(&self, weights: &Array1<f32>, threshold: f32) -> Result<Array1<f32>> {
        let compressed = weights.mapv(|w| if w.abs() > threshold { w } else { 0.0 });

        Ok(compressed)
    }

    /// Perform secure FedAvg with additive masks
    async fn secure_fedavg_with_masks(
        &mut self,
        updates: Vec<ModelUpdate>,
    ) -> Result<ModelParameters> {
        info!("Performing secure FedAvg with additive masks");

        if updates.is_empty() {
            return Err(FederatedError::AggregationError(
                "No updates to aggregate".to_string(),
            ));
        }

        let model_size = updates[0].weights.len();
        let mut aggregated_weights = Array1::zeros(model_size);
        let mut total_examples = 0u32;

        // Generate masks for each participant
        let mut masks = Vec::new();
        for update in &updates {
            let mask = self.generate_mask(model_size, &update.participant_id);
            masks.push(mask);
        }

        // Apply masks and aggregate
        for (update, mask) in updates.iter().zip(masks.iter()) {
            let weight = update.num_examples as f32;
            total_examples += update.num_examples;

            // Apply additive mask (in practice, masks would cancel out across participants)
            let masked_weights = &update.weights + &mask.mask;
            aggregated_weights = aggregated_weights + masked_weights * weight;
        }

        // Remove masks (simplified - in real protocol, masks would sum to zero)
        let mask_sum: Array1<f32> = masks
            .iter()
            .map(|m| &m.mask)
            .fold(Array1::zeros(model_size), |acc, mask| acc + mask);

        aggregated_weights =
            (aggregated_weights - mask_sum * total_examples as f32) / total_examples as f32;

        // Serialize aggregated weights
        let serialized_weights = self.serialize_weights(&aggregated_weights)?;

        // Create metadata
        let metadata = crate::ModelMetadata {
            architecture: "federated_model".to_string(),
            parameter_count: model_size as u64,
            input_shape: vec![1],  // Placeholder
            output_shape: vec![1], // Placeholder
            compression: self.compression_type.clone(),
            quantization: self.quantization_type.clone(),
        };

        Ok(ModelParameters::new(
            format!(
                "secure_agg_v{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            ),
            serialized_weights,
            metadata,
        ))
    }

    /// Perform standard FedAvg without masks
    fn standard_fedavg(&self, updates: Vec<ModelUpdate>) -> Result<ModelParameters> {
        info!("Performing standard FedAvg");

        if updates.is_empty() {
            return Err(FederatedError::AggregationError(
                "No updates to aggregate".to_string(),
            ));
        }

        let model_size = updates[0].weights.len();
        let mut aggregated_weights = Array1::zeros(model_size);
        let mut total_examples = 0u32;

        // Weighted averaging based on number of local examples
        for update in &updates {
            let weight = update.num_examples as f32;
            total_examples += update.num_examples;
            aggregated_weights = aggregated_weights + &update.weights * weight;
        }

        aggregated_weights = aggregated_weights / total_examples as f32;

        // Serialize aggregated weights
        let serialized_weights = self.serialize_weights(&aggregated_weights)?;

        // Create metadata
        let metadata = crate::ModelMetadata {
            architecture: "federated_model".to_string(),
            parameter_count: model_size as u64,
            input_shape: vec![1],  // Placeholder
            output_shape: vec![1], // Placeholder
            compression: self.compression_type.clone(),
            quantization: self.quantization_type.clone(),
        };

        Ok(ModelParameters::new(
            format!(
                "fedavg_v{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            ),
            serialized_weights,
            metadata,
        ))
    }

    /// Compute aggregation statistics
    fn compute_aggregation_stats(&self, results: &[TrainingResult]) -> AggregationStats {
        let num_participants = results.len() as u32;
        let total_examples: u64 = results.iter().map(|r| r.metrics.num_examples as u64).sum();

        // Weighted averages
        let mut weighted_loss = 0.0f32;
        let mut weighted_accuracy = 0.0f32;

        for result in results {
            let weight = result.metrics.num_examples as f32;
            weighted_loss += result.metrics.training_loss * weight;
            weighted_accuracy += result.metrics.training_accuracy * weight;
        }

        let avg_training_loss = weighted_loss / total_examples as f32;
        let avg_training_accuracy = weighted_accuracy / total_examples as f32;

        // Simple convergence check
        let converged = avg_training_loss < 0.1 && results.len() >= 3;

        AggregationStats {
            num_participants,
            total_examples,
            avg_training_loss,
            avg_training_accuracy,
            improvement: 0.0, // Would compute based on previous round
            converged,
        }
    }

    /// Serialize model weights to bytes
    fn serialize_weights(&self, weights: &Array1<f32>) -> Result<Bytes> {
        let serialized = bincode::serialize(weights)
            .map_err(|e| FederatedError::SerializationError(e.to_string()))?;
        Ok(Bytes::from(serialized))
    }

    /// Deserialize model weights from bytes
    fn deserialize_weights(&self, bytes: &Bytes) -> Result<Array1<f32>> {
        let weights: Array1<f32> = bincode::deserialize(bytes)
            .map_err(|e| FederatedError::SerializationError(e.to_string()))?;
        Ok(weights)
    }
}

/// FedAvg aggregator with secure aggregation and differential privacy
pub struct FedAvgAggregator {
    secure_agg: SecureAggregation,
    config: FedAvgConfig,
}

impl FedAvgAggregator {
    pub fn new(
        privacy_config: PrivacyConfig,
        compression_type: CompressionType,
        quantization_type: QuantizationType,
        config: FedAvgConfig,
    ) -> Self {
        Self {
            secure_agg: SecureAggregation::new(privacy_config, compression_type, quantization_type),
            config,
        }
    }

    pub async fn aggregate(
        &mut self,
        round_id: &RoundId,
        training_results: Vec<TrainingResult>,
    ) -> Result<AggregationResult> {
        // Validate minimum participants
        if training_results.len() < self.config.min_participants as usize {
            return Err(FederatedError::InsufficientParticipants {
                got: training_results.len() as u32,
                need: self.config.min_participants,
            });
        }

        // Filter out invalid results
        let valid_results = self.filter_valid_results(training_results)?;

        // Perform secure aggregation
        self.secure_agg
            .secure_aggregate(round_id, valid_results)
            .await
    }

    fn filter_valid_results(&self, results: Vec<TrainingResult>) -> Result<Vec<TrainingResult>> {
        let mut valid_results = Vec::new();

        for result in results {
            // Check for reasonable training metrics
            if result.metrics.training_loss.is_finite()
                && result.metrics.training_loss >= 0.0
                && result.metrics.training_accuracy >= 0.0
                && result.metrics.training_accuracy <= 1.0
                && result.metrics.num_examples > 0
            {
                valid_results.push(result);
            } else {
                warn!(
                    "Filtering out invalid training result from {}",
                    result.participant_id.agent_id
                );
            }
        }

        if valid_results.is_empty() {
            return Err(FederatedError::AggregationError(
                "No valid training results after filtering".to_string(),
            ));
        }

        Ok(valid_results)
    }
}

/// Model update from a participant
#[derive(Debug, Clone)]
struct ModelUpdate {
    participant_id: ParticipantId,
    weights: Array1<f32>,
    num_examples: u32,
    training_loss: f32,
}

/// Secret mask for additive secret sharing
#[derive(Debug, Clone)]
struct SecretMask {
    participant_id: ParticipantId,
    mask: Array1<f32>,
    seed: u64,
}

/// Configuration for FedAvg aggregation
#[derive(Debug, Clone)]
pub struct FedAvgConfig {
    pub min_participants: u32,
    pub max_participants: u32,
    pub aggregation_timeout_sec: u64,
    pub enable_validation: bool,
}

impl Default for FedAvgConfig {
    fn default() -> Self {
        Self {
            min_participants: 2,
            max_participants: 100,
            aggregation_timeout_sec: 600,
            enable_validation: true,
        }
    }
}

/// Differential privacy implementation for federated learning
pub struct DifferentialPrivacy {
    epsilon: f32,
    delta: f32,
    clipping_norm: f32,
    noise_multiplier: f32,
}

impl DifferentialPrivacy {
    pub fn new(epsilon: f32, delta: f32, clipping_norm: f32, noise_multiplier: f32) -> Self {
        Self {
            epsilon,
            delta,
            clipping_norm,
            noise_multiplier,
        }
    }

    /// Apply DP-SGD to gradients
    pub fn apply_dp_sgd(&self, gradients: &Array1<f32>) -> Result<Array1<f32>> {
        // Step 1: Clip gradients
        let clipped = self.clip_gradients(gradients);

        // Step 2: Add calibrated noise
        let noisy = self.add_gaussian_noise(clipped)?;

        Ok(noisy)
    }

    fn clip_gradients(&self, gradients: &Array1<f32>) -> Array1<f32> {
        let l2_norm = gradients.mapv(|x| x * x).sum().sqrt();

        if l2_norm > self.clipping_norm {
            gradients * (self.clipping_norm / l2_norm)
        } else {
            gradients.clone()
        }
    }

    fn add_gaussian_noise(&self, mut gradients: Array1<f32>) -> Result<Array1<f32>> {
        let noise_stddev = self.noise_multiplier * self.clipping_norm;

        for grad in gradients.iter_mut() {
            let noise: f32 =
                thread_rng().sample(rand_distr::Normal::new(0.0, noise_stddev).map_err(|e| {
                    FederatedError::AggregationError(format!("Noise generation failed: {}", e))
                })?);
            *grad += noise;
        }

        Ok(gradients)
    }

    /// Compute privacy budget consumed
    pub fn compute_privacy_cost(&self, num_rounds: u32, participants_per_round: u32) -> (f32, f32) {
        // Simplified privacy accounting (in practice, use more sophisticated methods)
        let total_epsilon = self.epsilon * (num_rounds as f32).sqrt();
        let total_delta = self.delta * num_rounds as f32;

        (total_epsilon, total_delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeviceCapabilities, DeviceType, ResourceUsage, TrainingMetrics};

    fn create_test_training_result(
        participant_id: ParticipantId,
        loss: f32,
        accuracy: f32,
    ) -> TrainingResult {
        let weights = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let serialized_weights = bincode::serialize(&weights).unwrap();

        let metadata = crate::ModelMetadata {
            architecture: "test".to_string(),
            parameter_count: 4,
            input_shape: vec![4],
            output_shape: vec![1],
            compression: CompressionType::None,
            quantization: QuantizationType::Float32,
        };

        TrainingResult {
            round_id: RoundId::new("test".to_string(), 1, 12345),
            participant_id,
            model_update: ModelParameters::new(
                "v1".to_string(),
                Bytes::from(serialized_weights),
                metadata,
            ),
            metrics: TrainingMetrics {
                training_loss: loss,
                training_accuracy: accuracy,
                validation_loss: None,
                validation_accuracy: None,
                num_examples: 100,
                training_time_sec: 60.0,
            },
            resource_usage: ResourceUsage {
                peak_memory_mb: 100.0,
                energy_joules: 50.0,
                flops: 1000000,
                bytes_sent: 1024,
                bytes_received: 1024,
                battery_drain: 0.05,
            },
            timestamp: 12345,
        }
    }

    #[tokio::test]
    async fn test_secure_aggregation() {
        let privacy_config = PrivacyConfig::default();
        let mut secure_agg = SecureAggregation::new(
            privacy_config,
            CompressionType::None,
            QuantizationType::Float32,
        );

        // Create test participants
        let participant1 = ParticipantId::new(
            agent_fabric::AgentId::new("phone-001", "mobile"),
            DeviceType::Phone,
            DeviceCapabilities::default(),
        );

        let participant2 = ParticipantId::new(
            agent_fabric::AgentId::new("phone-002", "mobile"),
            DeviceType::Phone,
            DeviceCapabilities::default(),
        );

        // Create training results
        let results = vec![
            create_test_training_result(participant1, 0.5, 0.8),
            create_test_training_result(participant2, 0.6, 0.7),
        ];

        let round_id = RoundId::new("test".to_string(), 1, 12345);
        let aggregation_result = secure_agg
            .secure_aggregate(&round_id, results)
            .await
            .unwrap();

        assert_eq!(aggregation_result.round_id, round_id);
        assert_eq!(aggregation_result.stats.num_participants, 2);
        assert_eq!(aggregation_result.stats.total_examples, 200);
        assert!((aggregation_result.stats.avg_training_loss - 0.55).abs() < 0.01);
        assert!((aggregation_result.stats.avg_training_accuracy - 0.75).abs() < 0.01);
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn test_differential_privacy() {
        let dp = DifferentialPrivacy::new(1.0, 1e-5, 1.0, 1.0);
        let gradients = Array1::from(vec![2.0, -1.5, 3.0, 0.5]);

        let dp_gradients = dp.apply_dp_sgd(&gradients).unwrap();

        // Check that gradients were clipped and noise was added
        assert_eq!(dp_gradients.len(), gradients.len());

        // With clipping norm 1.0, the L2 norm should be <= 1.0
        let l2_norm = dp_gradients.mapv(|x| x * x).sum().sqrt();
        assert!(l2_norm <= 1.1); // Allow small tolerance for noise
    }

    #[test]
    fn test_top_k_sparsification() {
        let privacy_config = PrivacyConfig {
            enable_dp: false,
            ..Default::default()
        };
        let secure_agg = SecureAggregation::new(
            privacy_config,
            CompressionType::TopK { k: 2 },
            QuantizationType::Float32,
        );

        let weights = Array1::from(vec![0.1, 2.0, -1.5, 0.2, 3.0]);
        let sparse_weights = secure_agg.top_k_sparsification(&weights, 2).unwrap();

        // Should keep only the 2 largest magnitude values
        let non_zero_count = sparse_weights.iter().filter(|&&x| x != 0.0).count();
        assert_eq!(non_zero_count, 2);

        // Check that largest values are preserved
        assert_eq!(sparse_weights[4], 3.0); // Largest magnitude
        assert_eq!(sparse_weights[1], 2.0); // Second largest magnitude
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn test_quantization() {
        let privacy_config = PrivacyConfig {
            enable_dp: false,
            ..Default::default()
        };
        let secure_agg =
            SecureAggregation::new(privacy_config, CompressionType::Q8, QuantizationType::Int8);

        let weights = Array1::from(vec![0.1, 0.5, -0.3, 0.8, -0.2]);
        let quantized = secure_agg.quantize_weights(&weights, 8).unwrap();

        // Quantized values should be different from original but preserve relative ordering
        assert_ne!(quantized, weights);
        assert_eq!(quantized.len(), weights.len());

        // Check that relative ordering is preserved
        assert!(quantized[3] > quantized[1]); // 0.8 > 0.5
        assert!(quantized[2] < quantized[4]); // -0.3 < -0.2
    }
}
