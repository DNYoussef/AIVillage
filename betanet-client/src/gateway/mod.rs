//! Gateway module for SCION-ish path-aware networking

pub mod scion_gateway;

pub use scion_gateway::{
    ScionGateway, PathInfo, PathRequirements, GatewayStatistics,
    ControlMessage, ControlMessageType, PathSelectionStrategy,
};
