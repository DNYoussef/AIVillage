//! Betanet FFI - C bindings for Betanet protocol components
//!
//! This library provides C-compatible FFI bindings for core Betanet functionality
//! For Day 8-9 deliverable, using minimal implementation to avoid OpenSSL issues on Windows

// Core shared types and utilities
mod common;
pub use common::*;

// Base library functions (initialization, buffers, etc.)
mod lib_minimal;
pub use lib_minimal::*;

// Protocol modules
mod htx;
pub use htx::*;

mod mixnode;
pub use mixnode::*;

mod utls;
pub use utls::*;

mod linter;
pub use linter::*;
