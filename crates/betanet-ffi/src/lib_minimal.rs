//! Betanet FFI - Minimal C bindings for Day 8-9 deliverable
//!
//! This is a minimal version for demonstrating FFI capabilities without OpenSSL dependencies

use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_uint};
use std::ptr;

/// Result codes for Betanet FFI functions
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BetanetResult {
    /// Operation completed successfully
    Success = 0,
    /// Invalid argument provided
    InvalidArgument = -1,
    /// Out of memory
    OutOfMemory = -2,
    /// Network error
    NetworkError = -3,
    /// Parsing error
    ParseError = -4,
    /// Cryptographic error
    CryptoError = -5,
    /// Internal error
    InternalError = -6,
    /// Feature not supported
    NotSupported = -7,
    /// Operation timed out
    Timeout = -8,
}

/// Buffer structure for passing data between C and Rust
#[repr(C)]
pub struct BetanetBuffer {
    /// Pointer to data
    pub data: *mut u8,
    /// Length of data in bytes
    pub len: c_uint,
    /// Capacity of buffer (for owned buffers)
    pub capacity: c_uint,
}

impl BetanetBuffer {
    /// Create a new empty buffer
    pub fn new() -> Self {
        Self {
            data: ptr::null_mut(),
            len: 0,
            capacity: 0,
        }
    }

    /// Create buffer from Vec<u8>
    pub fn from_vec(mut vec: Vec<u8>) -> Self {
        let data = vec.as_mut_ptr();
        let len = vec.len() as c_uint;
        let capacity = vec.capacity() as c_uint;
        std::mem::forget(vec); // Transfer ownership to C

        Self {
            data,
            len,
            capacity,
        }
    }

    /// Create buffer from slice (read-only)
    pub fn from_slice(slice: &[u8]) -> Self {
        Self {
            data: slice.as_ptr() as *mut u8,
            len: slice.len() as c_uint,
            capacity: 0, // Indicates read-only
        }
    }

    /// Convert back to Vec<u8> (takes ownership)
    ///
    /// # Safety
    /// Buffer must have been created from a Vec with from_vec()
    pub unsafe fn into_vec(self) -> Vec<u8> {
        if self.capacity == 0 || self.data.is_null() {
            return Vec::new();
        }

        Vec::from_raw_parts(self.data, self.len as usize, self.capacity as usize)
    }

    /// Get slice view of buffer data
    ///
    /// # Safety
    /// Buffer data pointer must be valid
    pub unsafe fn as_slice(&self) -> &[u8] {
        if self.data.is_null() || self.len == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(self.data, self.len as usize)
        }
    }
}

/// Initialize the Betanet library
///
/// Must be called before using any other functions.
/// Returns 0 on success, negative error code on failure.
#[no_mangle]
pub extern "C" fn betanet_init() -> c_int {
    // Initialize logging if needed
    // Initialize any global state
    0
}

/// Cleanup and shutdown the Betanet library
///
/// Should be called when finished using the library.
#[no_mangle]
pub extern "C" fn betanet_cleanup() {
    // Cleanup any global state
}

/// Get library version string
///
/// Returns a null-terminated string with the library version.
/// The caller must not free the returned pointer.
#[no_mangle]
pub extern "C" fn betanet_version() -> *const c_char {
    static VERSION: &[u8] = b"0.1.0-minimal\0";
    VERSION.as_ptr() as *const c_char
}

/// Check if a feature is supported
///
/// # Arguments
/// * `feature` - Feature name to check (null-terminated string)
///
/// # Returns
/// * `BetanetResult::Success` if feature is supported
/// * `BetanetResult::NotSupported` if feature is not supported
/// * `BetanetResult::InvalidArgument` if feature name is invalid
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
        "ffi_demo" | "buffer_management" | "version_info" => BetanetResult::Success,
        _ => BetanetResult::NotSupported,
    }
}

/// Free a buffer allocated by Betanet
///
/// # Arguments
/// * `buffer` - Buffer to free
///
/// # Safety
/// Buffer must have been allocated by Betanet library
#[no_mangle]
pub unsafe extern "C" fn betanet_buffer_free(buffer: BetanetBuffer) {
    if buffer.capacity > 0 && !buffer.data.is_null() {
        // This was an owned buffer, convert back to Vec to drop
        let _vec = buffer.into_vec();
    }
}

/// Allocate a new buffer
///
/// # Arguments
/// * `size` - Size in bytes to allocate
///
/// # Returns
/// * New buffer on success
/// * Empty buffer on failure
#[no_mangle]
pub extern "C" fn betanet_buffer_alloc(size: c_uint) -> BetanetBuffer {
    if size == 0 {
        return BetanetBuffer::new();
    }

    let vec = vec![0u8; size as usize];
    BetanetBuffer::from_vec(vec)
}

/// Get error message for result code
///
/// # Arguments
/// * `result` - Result code
///
/// # Returns
/// * Null-terminated error message string
/// * The caller must not free the returned pointer
#[no_mangle]
pub extern "C" fn betanet_error_message(result: BetanetResult) -> *const c_char {
    let msg = match result {
        BetanetResult::Success => "Success\0",
        BetanetResult::InvalidArgument => "Invalid argument\0",
        BetanetResult::OutOfMemory => "Out of memory\0",
        BetanetResult::NetworkError => "Network error\0",
        BetanetResult::ParseError => "Parse error\0",
        BetanetResult::CryptoError => "Cryptographic error\0",
        BetanetResult::InternalError => "Internal error\0",
        BetanetResult::NotSupported => "Feature not supported\0",
        BetanetResult::Timeout => "Operation timed out\0",
    };

    msg.as_ptr() as *const c_char
}

/// Simple packet encoder/decoder demo
///
/// # Arguments
/// * `input` - Input data
/// * `output` - Output buffer (will be allocated)
///
/// # Returns
/// * Result code
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

    // Simple demo encoding: add length prefix
    let mut encoded = Vec::with_capacity(4 + input_data.len());
    encoded.extend_from_slice(&(input_data.len() as u32).to_be_bytes());
    encoded.extend_from_slice(input_data);

    unsafe {
        *output = BetanetBuffer::from_vec(encoded);
    }

    BetanetResult::Success
}

/// Simple packet decoder demo
///
/// # Arguments
/// * `input` - Input encoded data
/// * `output` - Output buffer (will be allocated)
///
/// # Returns
/// * Result code
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

    // Decode length prefix
    let length =
        u32::from_be_bytes([input_data[0], input_data[1], input_data[2], input_data[3]]) as usize;

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
///
/// # Arguments
/// * `input` - Input string (null-terminated)
/// * `output` - Output buffer (will be allocated)
///
/// # Returns
/// * Result code
#[no_mangle]
pub extern "C" fn betanet_echo(input: *const c_char, output: *mut BetanetBuffer) -> BetanetResult {
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
