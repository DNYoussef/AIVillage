//! Betanet FFI - Core library functions
//!
//! Provides basic initialization and demo packet utilities using
//! shared types from [`common`].

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

use crate::common::{BetanetBuffer, BetanetResult};

/// Initialize the Betanet library
///
/// Must be called before using any other functions.
#[no_mangle]
pub extern "C" fn betanet_init() -> c_int {
    // Initialize logging or global state here if necessary
    0
}

/// Cleanup and shutdown the Betanet library
#[no_mangle]
pub extern "C" fn betanet_cleanup() {
    // Cleanup any global state here if necessary
}

/// Get library version string
///
/// Returns a null-terminated string with the library version.
/// The caller must not free the returned pointer.
#[no_mangle]
pub extern "C" fn betanet_version() -> *const c_char {
    static VERSION: &[u8] = b"0.1.0\0";
    VERSION.as_ptr() as *const c_char
}

/// Check if a feature is supported
///
/// * `feature` - Feature name to check (null-terminated string)
///
/// Returns [`BetanetResult::Success`] if supported,
/// [`BetanetResult::NotSupported`] if not, or
/// [`BetanetResult::InvalidArgument`] on error.
#[no_mangle]
pub extern "C" fn betanet_feature_supported(feature: *const c_char) -> BetanetResult {
    if feature.is_null() {
        return BetanetResult::InvalidArgument;
    }

    let feature_cstr = unsafe { CStr::from_ptr(feature) };
    let feature_str = match feature_cstr.to_str() {
        Ok(s) => s,
        Err(_) => return BetanetResult::InvalidArgument,
    };

    match feature_str {
        "ffi_demo" | "buffer_management" | "version_info" |
        "htx" | "mixnode" | "sphinx" | "utls" | "ja3" | "ja4" | "linter" => BetanetResult::Success,
        _ => BetanetResult::NotSupported,
    }
}

/// Simple packet encoder demo
#[no_mangle]
pub extern "C" fn betanet_packet_encode(
    input: BetanetBuffer,
    output: *mut BetanetBuffer,
) -> BetanetResult {
    if output.is_null() {
        return BetanetResult::InvalidArgument;
    }

    let input_data = unsafe {
        if input.data.is_null() {
            &[]
        } else {
            input.as_slice()
        }
    };

    let mut encoded = Vec::with_capacity(4 + input_data.len());
    encoded.extend_from_slice(&(input_data.len() as u32).to_be_bytes());
    encoded.extend_from_slice(input_data);

    unsafe {
        *output = BetanetBuffer::from_vec(encoded);
    }

    BetanetResult::Success
}

/// Simple packet decoder demo
#[no_mangle]
pub extern "C" fn betanet_packet_decode(
    input: BetanetBuffer,
    output: *mut BetanetBuffer,
) -> BetanetResult {
    if output.is_null() {
        return BetanetResult::InvalidArgument;
    }

    let input_data = unsafe {
        if input.data.is_null() {
            return BetanetResult::InvalidArgument;
        }
        input.as_slice()
    };

    if input_data.len() < 4 {
        return BetanetResult::ParseError;
    }

    let length = u32::from_be_bytes([
        input_data[0],
        input_data[1],
        input_data[2],
        input_data[3],
    ]) as usize;

    if input_data.len() < 4 + length {
        return BetanetResult::ParseError;
    }

    let payload = input_data[4..4 + length].to_vec();

    unsafe {
        *output = BetanetBuffer::from_vec(payload);
    }

    BetanetResult::Success
}

/// Echo function for testing
#[no_mangle]
pub extern "C" fn betanet_echo(
    input: *const c_char,
    output: *mut BetanetBuffer,
) -> BetanetResult {
    if input.is_null() || output.is_null() {
        return BetanetResult::InvalidArgument;
    }

    let input_cstr = unsafe { CStr::from_ptr(input) };
    let input_str = match input_cstr.to_str() {
        Ok(s) => s,
        Err(_) => return BetanetResult::InvalidArgument,
    };

    let echo_msg = format!("Echo: {}", input_str);
    let echo_bytes = echo_msg.into_bytes();

    unsafe {
        *output = BetanetBuffer::from_vec(echo_bytes);
    }

    BetanetResult::Success
}
