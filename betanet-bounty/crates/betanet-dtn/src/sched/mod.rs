//! DTN Queue Scheduling and Stability Control
//!
//! This module implements queue-aware scheduling algorithms for DTN bundle transmission
//! to optimize delivery performance under energy and privacy constraints.

pub mod lyapunov;
pub mod synthetic_tests;
pub mod performance_tests;

pub use lyapunov::{LyapunovScheduler, LyapunovConfig, SchedulingDecision, QueueState};
pub use synthetic_tests::{SyntheticContactGenerator, FifoScheduler, TestResults};
pub use performance_tests::{PerformanceTestFramework, SchedulerTestSuite, TopologyType};
