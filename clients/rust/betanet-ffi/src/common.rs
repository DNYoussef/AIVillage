//! Common FFI types and utilities

use std::os::raw::c_char;
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

/// Opaque handle for Betanet objects
#[repr(C)]
pub struct BetanetHandle {
    _private: [u8; 0],
}

/// Buffer structure for passing data between C and Rust
#[repr(C)]
pub struct BetanetBuffer {
    /// Pointer to data
    pub data: *mut u8,
    /// Length of data in bytes
    pub len: usize,
    /// Capacity of buffer (for owned buffers)
    pub capacity: usize,
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
        let len = vec.len();
        let capacity = vec.capacity();
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
            len: slice.len(),
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

        Vec::from_raw_parts(self.data, self.len, self.capacity)
    }

    /// Get slice view of buffer data
    ///
    /// # Safety
    /// Buffer data pointer must be valid
    pub unsafe fn as_slice(&self) -> &[u8] {
        if self.data.is_null() || self.len == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(self.data, self.len)
        }
    }
}

/// Free a buffer allocated by Betanet
///
/// This function must be called by C callers to release buffers returned by
/// the Betanet library. Failing to do so or attempting to free the pointer with
/// another method will result in memory leaks.
///
/// # Arguments
/// * `buffer` - Buffer to free
///
/// # Safety
/// Buffer must have been allocated by Betanet library
#[no_mangle]
pub unsafe extern "C" fn betanet_buffer_free(buffer: BetanetBuffer) {
    if buffer.capacity > 0 && !buffer.data.is_null() {
        // Overwrite buffer contents before dropping
        ptr::write_bytes(buffer.data, 0, buffer.len);

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
pub extern "C" fn betanet_buffer_alloc(size: usize) -> BetanetBuffer {
    if size == 0 {
        return BetanetBuffer::new();
    }

    let vec = vec![0u8; size];
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
