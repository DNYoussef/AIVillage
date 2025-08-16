//! Betanet FFI - C bindings for Betanet protocol components
//!
//! This library provides C-compatible FFI bindings for core Betanet functionality
//! For Day 8-9 deliverable, using minimal implementation to avoid OpenSSL issues on Windows

// Use minimal implementation for now to avoid OpenSSL build issues
mod lib_minimal;
pub use lib_minimal::*;

// Full implementation (commented out due to OpenSSL dependency issues)
// mod htx;
// mod mixnode;
// mod utls;
// mod linter;
// mod common;
