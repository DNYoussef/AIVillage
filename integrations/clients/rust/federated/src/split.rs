//! Split Learning Implementation
//!
//! Enables split learning where early layers run on device and later layers
//! run on beacon nodes, with record/replay of microbatches.

use crate::{FederatedError, Result};

/// Split learning coordinator which connects device side computation with a
/// beacon for completing the remaining layers of the model.  The coordinator
/// manages micro-batch forwarding and supports replay of recorded activations
/// for fault tolerance.
pub struct SplitLearning {
    device: DeviceTraining,
    beacon: BeaconAggregation,
}

impl SplitLearning {
    /// Create a new split learning instance
    pub fn new() -> Self {
        Self {
            device: DeviceTraining::new(),
            beacon: BeaconAggregation::new(),
        }
    }

    /// Process a series of microbatches, returning the aggregated result from
    /// the beacon.  Each microbatch is recorded for potential replay.
    pub fn train_round(&mut self, microbatches: Vec<Vec<f32>>) -> Result<Vec<f32>> {
        let mut activations = Vec::new();
        for batch in microbatches {
            activations.push(self.device.forward(batch));
        }
        self.beacon.aggregate(&activations)
    }

    /// Retrieve a copy of recorded microbatches for replay
    pub fn replay_microbatches(&self) -> Result<Vec<Vec<f32>>> {
        Ok(self.device.replay())
    }
}

/// Device-side training for split learning.  Stores the output activations of
/// early layers for replay in case of failures.
pub struct DeviceTraining {
    replay_buffer: Vec<Vec<f32>>,
}

impl DeviceTraining {
    pub fn new() -> Self {
        Self {
            replay_buffer: Vec::new(),
        }
    }

    /// Forward a microbatch through the device portion of the model.  For this
    /// simplified example the forward pass is identity; the microbatch is stored
    /// for replay and returned unchanged.
    pub fn forward(&mut self, batch: Vec<f32>) -> Vec<f32> {
        self.replay_buffer.push(batch.clone());
        batch
    }

    /// Return all recorded microbatches
    fn replay(&self) -> Vec<Vec<f32>> {
        self.replay_buffer.clone()
    }
}

/// Beacon aggregation for split learning.  Aggregates activations received from
/// devices and produces a final output.
pub struct BeaconAggregation;

impl BeaconAggregation {
    pub fn new() -> Self {
        Self {}
    }

    /// Aggregate activations by computing their element-wise mean
    pub fn aggregate(&self, activations: &[Vec<f32>]) -> Result<Vec<f32>> {
        if activations.is_empty() {
            return Err(FederatedError::AggregationError(
                "no activations".into(),
            ));
        }
        let dim = activations[0].len();
        let mut sum = vec![0.0; dim];
        for act in activations {
            for (i, v) in act.iter().enumerate() {
                sum[i] += v;
            }
        }
        let n = activations.len() as f32;
        Ok(sum.into_iter().map(|v| v / n).collect())
    }
}
