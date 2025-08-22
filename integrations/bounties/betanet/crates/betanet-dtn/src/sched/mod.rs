//! DTN Queue Scheduling and Stability Control
//!
//! This module implements queue-aware scheduling algorithms for DTN bundle transmission
//! to optimize delivery performance under energy and privacy constraints.

pub mod lyapunov;
pub mod performance_tests;
pub mod synthetic_tests;

pub use lyapunov::{LyapunovConfig, LyapunovScheduler, QueueState, SchedulingDecision};
pub use performance_tests::{PerformanceTestFramework, SchedulerTestSuite, TopologyType};
pub use synthetic_tests::{FifoScheduler, SyntheticContactGenerator, TestResults};
