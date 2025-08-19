# Rust Client Infrastructure - FFI Integration

## FFI Architecture Overview

The AIVillage Rust Client Infrastructure provides Foreign Function Interface (FFI) bindings to enable seamless integration across multiple programming languages and platforms. The FFI layer abstracts the Rust implementation while maintaining performance and security guarantees.

## FFI Design Principles

### Memory Safety
- **No Raw Pointers**: All data transfer uses opaque handles and structured data
- **RAII Pattern**: Automatic resource cleanup through drop handlers
- **Bounds Checking**: All array/string operations include length validation
- **Exception Safety**: C-compatible error handling with result codes

### API Consistency
- **Unified Interface**: Consistent function naming and parameter patterns
- **Type Safety**: Strong typing with clear ownership semantics
- **Version Compatibility**: ABI-stable interface with versioned symbols
- **Platform Agnostic**: Consistent behavior across Windows, macOS, Linux

## C FFI Implementation

### Core Types and Structures

#### Opaque Handle Types
**File**: `crates/betanet-ffi/src/types.rs`

```rust
// Opaque handles to prevent direct access to Rust types
#[repr(C)]
pub struct HtxSessionHandle {
    _private: [u8; 0],
}

#[repr(C)]
pub struct MixnodeHandle {
    _private: [u8; 0],
}

#[repr(C)]
pub struct AgentFabricHandle {
    _private: [u8; 0],
}

// C-compatible result type
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FFIResult {
    Success = 0,
    InvalidHandle = -1,
    InvalidInput = -2,
    NetworkError = -3,
    CryptoError = -4,
    OutOfMemory = -5,
    Timeout = -6,
    PermissionDenied = -7,
}

// Buffer management for data transfer
#[repr(C)]
pub struct FFIBuffer {
    pub data: *mut u8,
    pub length: usize,
    pub capacity: usize,
}

impl FFIBuffer {
    pub fn from_vec(mut vec: Vec<u8>) -> Self {
        let data = vec.as_mut_ptr();
        let length = vec.len();
        let capacity = vec.capacity();
        std::mem::forget(vec); // Prevent deallocation

        Self { data, length, capacity }
    }

    pub unsafe fn to_vec(self) -> Vec<u8> {
        Vec::from_raw_parts(self.data, self.length, self.capacity)
    }
}
```

#### Configuration Structures
```rust
#[repr(C)]
pub struct HtxConfig {
    pub listen_addr: *const c_char,
    pub enable_tcp: bool,
    pub enable_quic: bool,
    pub enable_noise_xk: bool,
    pub max_connections: u32,
    pub connection_timeout_secs: u64,
    pub static_private_key: *const u8,
    pub static_private_key_len: usize,
}

#[repr(C)]
pub struct MixnodeConfig {
    pub node_id: *const c_char,
    pub listen_addr: *const c_char,
    pub mix_strategy: u32, // 0=threshold, 1=timed, 2=binomial
    pub mix_threshold: u32,
    pub min_delay_ms: u64,
    pub max_delay_ms: u64,
    pub enable_cover_traffic: bool,
}
```

### Core API Functions

#### HTX Transport API
**File**: `crates/betanet-ffi/src/htx.rs`

```rust
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;

/// Create new HTX session
#[no_mangle]
pub extern "C" fn htx_session_new(
    config: *const HtxConfig,
    is_initiator: bool,
    out_handle: *mut *mut HtxSessionHandle,
) -> FFIResult {
    if config.is_null() || out_handle.is_null() {
        return FFIResult::InvalidInput;
    }

    let config = unsafe { &*config };

    // Convert C strings to Rust strings
    let listen_addr = match unsafe { CStr::from_ptr(config.listen_addr) }.to_str() {
        Ok(addr) => addr,
        Err(_) => return FFIResult::InvalidInput,
    };

    // Extract private key if provided
    let private_key = if !config.static_private_key.is_null() && config.static_private_key_len == 32 {
        Some(unsafe {
            std::slice::from_raw_parts(config.static_private_key, 32)
        })
    } else {
        None
    };

    // Create Rust HTX configuration
    let rust_config = betanet_htx::HtxConfig {
        listen_addr: listen_addr.parse().map_err(|_| FFIResult::InvalidInput)?,
        enable_tcp: config.enable_tcp,
        enable_quic: config.enable_quic,
        enable_noise_xk: config.enable_noise_xk,
        max_connections: config.max_connections as usize,
        connection_timeout_secs: config.connection_timeout_secs,
        static_private_key: private_key.map(|k| bytes::Bytes::copy_from_slice(k)),
        ..Default::default()
    };

    // Create session
    match betanet_htx::HtxSession::new(rust_config, is_initiator) {
        Ok(session) => {
            let boxed_session = Box::new(session);
            unsafe {
                *out_handle = Box::into_raw(boxed_session) as *mut HtxSessionHandle;
            }
            FFIResult::Success
        }
        Err(_) => FFIResult::CryptoError,
    }
}

/// Send data on HTX stream
#[no_mangle]
pub extern "C" fn htx_session_send_data(
    handle: *mut HtxSessionHandle,
    stream_id: u32,
    data: *const u8,
    data_len: usize,
    out_buffer: *mut FFIBuffer,
) -> FFIResult {
    if handle.is_null() || data.is_null() || out_buffer.is_null() {
        return FFIResult::InvalidHandle;
    }

    let session = unsafe { &mut *(handle as *mut betanet_htx::HtxSession) };
    let input_data = unsafe { std::slice::from_raw_parts(data, data_len) };

    // Use async runtime for async function
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return FFIResult::OutOfMemory,
    };

    match rt.block_on(session.send_data(stream_id, input_data)) {
        Ok(encrypted_data) => {
            let buffer = FFIBuffer::from_vec(encrypted_data.to_vec());
            unsafe { *out_buffer = buffer; }
            FFIResult::Success
        }
        Err(_) => FFIResult::NetworkError,
    }
}

/// Process incoming HTX data
#[no_mangle]
pub extern "C" fn htx_session_process_data(
    handle: *mut HtxSessionHandle,
    data: *const u8,
    data_len: usize,
    out_messages: *mut *mut FFIStreamMessage,
    out_count: *mut usize,
) -> FFIResult {
    if handle.is_null() || data.is_null() || out_messages.is_null() || out_count.is_null() {
        return FFIResult::InvalidHandle;
    }

    let session = unsafe { &mut *(handle as *mut betanet_htx::HtxSession) };
    let input_data = unsafe { std::slice::from_raw_parts(data, data_len) };

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return FFIResult::OutOfMemory,
    };

    match rt.block_on(session.process_data(input_data)) {
        Ok(stream_data) => {
            let messages: Vec<FFIStreamMessage> = stream_data
                .into_iter()
                .map(|(stream_id, data)| FFIStreamMessage {
                    stream_id,
                    data: FFIBuffer::from_vec(data.to_vec()),
                })
                .collect();

            let count = messages.len();
            let boxed_messages = messages.into_boxed_slice();

            unsafe {
                *out_messages = Box::into_raw(boxed_messages) as *mut FFIStreamMessage;
                *out_count = count;
            }

            FFIResult::Success
        }
        Err(_) => FFIResult::NetworkError,
    }
}

/// Free HTX session
#[no_mangle]
pub extern "C" fn htx_session_free(handle: *mut HtxSessionHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle as *mut betanet_htx::HtxSession);
        }
    }
}
```

#### Mixnode API
**File**: `crates/betanet-ffi/src/mixnode.rs`

```rust
/// Create new mixnode
#[no_mangle]
pub extern "C" fn mixnode_new(
    config: *const MixnodeConfig,
    out_handle: *mut *mut MixnodeHandle,
) -> FFIResult {
    if config.is_null() || out_handle.is_null() {
        return FFIResult::InvalidInput;
    }

    let config = unsafe { &*config };

    // Convert C strings
    let node_id = match unsafe { CStr::from_ptr(config.node_id) }.to_str() {
        Ok(id) => id.to_string(),
        Err(_) => return FFIResult::InvalidInput,
    };

    let listen_addr = match unsafe { CStr::from_ptr(config.listen_addr) }.to_str() {
        Ok(addr) => addr,
        Err(_) => return FFIResult::InvalidInput,
    };

    // Create mixnode configuration
    let rust_config = betanet_mixnode::MixnodeConfig {
        node_id,
        listen_addr: listen_addr.parse().map_err(|_| FFIResult::InvalidInput)?,
        mix_strategy: match config.mix_strategy {
            0 => betanet_mixnode::MixStrategy::Threshold(config.mix_threshold),
            1 => betanet_mixnode::MixStrategy::Timed(Duration::from_millis(config.mix_threshold as u64)),
            2 => betanet_mixnode::MixStrategy::Binomial(config.mix_threshold as f64 / 100.0),
            _ => return FFIResult::InvalidInput,
        },
        delay_params: betanet_mixnode::DelayParameters {
            min_delay_ms: config.min_delay_ms,
            max_delay_ms: config.max_delay_ms,
        },
        enable_cover_traffic: config.enable_cover_traffic,
    };

    // Create mixnode instance
    match betanet_mixnode::SphinxMixnode::new(rust_config) {
        Ok(mixnode) => {
            let boxed_mixnode = Box::new(mixnode);
            unsafe {
                *out_handle = Box::into_raw(boxed_mixnode) as *mut MixnodeHandle;
            }
            FFIResult::Success
        }
        Err(_) => FFIResult::CryptoError,
    }
}

/// Process packet through mixnode
#[no_mangle]
pub extern "C" fn mixnode_process_packet(
    handle: *mut MixnodeHandle,
    packet: *const u8,
    packet_len: usize,
    out_buffer: *mut FFIBuffer,
) -> FFIResult {
    if handle.is_null() || packet.is_null() || out_buffer.is_null() {
        return FFIResult::InvalidHandle;
    }

    if packet_len != 2048 {
        return FFIResult::InvalidInput; // Sphinx packets are fixed size
    }

    let mixnode = unsafe { &mut *(handle as *mut betanet_mixnode::SphinxMixnode) };
    let input_packet = unsafe { std::slice::from_raw_parts(packet, packet_len) };

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return FFIResult::OutOfMemory,
    };

    match rt.block_on(mixnode.process_packet(input_packet)) {
        Ok(Some(output_packet)) => {
            let buffer = FFIBuffer::from_vec(output_packet);
            unsafe { *out_buffer = buffer; }
            FFIResult::Success
        }
        Ok(None) => {
            // Packet delivered to final destination
            unsafe {
                *out_buffer = FFIBuffer { data: ptr::null_mut(), length: 0, capacity: 0 };
            }
            FFIResult::Success
        }
        Err(_) => FFIResult::CryptoError,
    }
}

/// Free mixnode
#[no_mangle]
pub extern "C" fn mixnode_free(handle: *mut MixnodeHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle as *mut betanet_mixnode::SphinxMixnode);
        }
    }
}
```

### Memory Management

#### Buffer Allocation and Deallocation
```rust
/// Allocate buffer for FFI usage
#[no_mangle]
pub extern "C" fn ffi_buffer_alloc(size: usize) -> FFIBuffer {
    let mut vec = Vec::with_capacity(size);
    vec.resize(size, 0);
    FFIBuffer::from_vec(vec)
}

/// Free FFI buffer
#[no_mangle]
pub extern "C" fn ffi_buffer_free(buffer: FFIBuffer) {
    if !buffer.data.is_null() {
        unsafe {
            let _ = buffer.to_vec(); // Automatically deallocated
        }
    }
}

/// Copy data from FFI buffer
#[no_mangle]
pub extern "C" fn ffi_buffer_copy_data(
    buffer: *const FFIBuffer,
    out_data: *mut u8,
    max_len: usize,
) -> usize {
    if buffer.is_null() || out_data.is_null() {
        return 0;
    }

    let buffer = unsafe { &*buffer };
    if buffer.data.is_null() || buffer.length == 0 {
        return 0;
    }

    let copy_len = std::cmp::min(buffer.length, max_len);
    unsafe {
        std::ptr::copy_nonoverlapping(buffer.data, out_data, copy_len);
    }

    copy_len
}
```

### Error Handling

#### Comprehensive Error Mapping
```rust
impl From<betanet_htx::HtxError> for FFIResult {
    fn from(error: betanet_htx::HtxError) -> Self {
        match error {
            betanet_htx::HtxError::Io(_) => FFIResult::NetworkError,
            betanet_htx::HtxError::Noise(_) => FFIResult::CryptoError,
            betanet_htx::HtxError::Protocol(_) => FFIResult::InvalidInput,
            betanet_htx::HtxError::Config(_) => FFIResult::InvalidInput,
            _ => FFIResult::NetworkError,
        }
    }
}

/// Get error message for result code
#[no_mangle]
pub extern "C" fn ffi_error_message(result: FFIResult) -> *const c_char {
    let message = match result {
        FFIResult::Success => "Operation completed successfully",
        FFIResult::InvalidHandle => "Invalid handle provided",
        FFIResult::InvalidInput => "Invalid input parameters",
        FFIResult::NetworkError => "Network operation failed",
        FFIResult::CryptoError => "Cryptographic operation failed",
        FFIResult::OutOfMemory => "Out of memory",
        FFIResult::Timeout => "Operation timed out",
        FFIResult::PermissionDenied => "Permission denied",
    };

    // Leak string to ensure C compatibility
    let c_string = CString::new(message).unwrap();
    c_string.into_raw()
}
```

## Python Integration

### Ctypes Bindings
**File**: `py/aivillage/p2p/betanet/ffi_bindings.py`

```python
import ctypes
import sys
import os
from ctypes import Structure, POINTER, c_char_p, c_uint32, c_uint64, c_bool, c_size_t, c_void_p, c_ubyte
from enum import IntEnum
from pathlib import Path

class FFIResult(IntEnum):
    SUCCESS = 0
    INVALID_HANDLE = -1
    INVALID_INPUT = -2
    NETWORK_ERROR = -3
    CRYPTO_ERROR = -4
    OUT_OF_MEMORY = -5
    TIMEOUT = -6
    PERMISSION_DENIED = -7

class FFIBuffer(Structure):
    _fields_ = [
        ("data", POINTER(c_ubyte)),
        ("length", c_size_t),
        ("capacity", c_size_t),
    ]

    def to_bytes(self) -> bytes:
        """Convert FFI buffer to Python bytes"""
        if self.data and self.length > 0:
            return ctypes.string_at(self.data, self.length)
        return b""

class HtxConfig(Structure):
    _fields_ = [
        ("listen_addr", c_char_p),
        ("enable_tcp", c_bool),
        ("enable_quic", c_bool),
        ("enable_noise_xk", c_bool),
        ("max_connections", c_uint32),
        ("connection_timeout_secs", c_uint64),
        ("static_private_key", POINTER(c_ubyte)),
        ("static_private_key_len", c_size_t),
    ]

class BetanetFFI:
    """Python wrapper for Betanet FFI library"""

    def __init__(self, lib_path: str = None):
        if lib_path is None:
            lib_path = self._find_library()

        self.lib = ctypes.CDLL(lib_path)
        self._setup_function_signatures()

    def _find_library(self) -> str:
        """Find the betanet FFI library"""
        possible_names = [
            "libbetanet_ffi.so",     # Linux
            "libbetanet_ffi.dylib",  # macOS
            "betanet_ffi.dll",       # Windows
        ]

        # Check relative to this module
        base_path = Path(__file__).parent.parent.parent.parent / "target" / "release"

        for name in possible_names:
            lib_path = base_path / name
            if lib_path.exists():
                return str(lib_path)

        # Check system paths
        for name in possible_names:
            try:
                return ctypes.util.find_library(name.split('.')[0])
            except:
                continue

        raise RuntimeError("Could not find betanet FFI library")

    def _setup_function_signatures(self):
        """Setup ctypes function signatures"""

        # HTX Session functions
        self.lib.htx_session_new.argtypes = [
            POINTER(HtxConfig), c_bool, POINTER(c_void_p)
        ]
        self.lib.htx_session_new.restype = ctypes.c_int

        self.lib.htx_session_send_data.argtypes = [
            c_void_p, c_uint32, POINTER(c_ubyte), c_size_t, POINTER(FFIBuffer)
        ]
        self.lib.htx_session_send_data.restype = ctypes.c_int

        self.lib.htx_session_free.argtypes = [c_void_p]
        self.lib.htx_session_free.restype = None

        # Buffer management
        self.lib.ffi_buffer_alloc.argtypes = [c_size_t]
        self.lib.ffi_buffer_alloc.restype = FFIBuffer

        self.lib.ffi_buffer_free.argtypes = [FFIBuffer]
        self.lib.ffi_buffer_free.restype = None

        # Error handling
        self.lib.ffi_error_message.argtypes = [ctypes.c_int]
        self.lib.ffi_error_message.restype = c_char_p

    def create_htx_session(self, config: dict, is_initiator: bool = True) -> int:
        """Create HTX session"""

        # Prepare configuration
        c_config = HtxConfig()
        c_config.listen_addr = config.get("listen_addr", "127.0.0.1:9000").encode()
        c_config.enable_tcp = config.get("enable_tcp", True)
        c_config.enable_quic = config.get("enable_quic", False)
        c_config.enable_noise_xk = config.get("enable_noise_xk", True)
        c_config.max_connections = config.get("max_connections", 1000)
        c_config.connection_timeout_secs = config.get("connection_timeout_secs", 30)

        # Handle private key if provided
        private_key = config.get("static_private_key")
        if private_key:
            key_array = (c_ubyte * 32)(*private_key)
            c_config.static_private_key = key_array
            c_config.static_private_key_len = 32
        else:
            c_config.static_private_key = None
            c_config.static_private_key_len = 0

        # Create session
        handle = c_void_p()
        result = self.lib.htx_session_new(
            ctypes.byref(c_config),
            is_initiator,
            ctypes.byref(handle)
        )

        if result != FFIResult.SUCCESS:
            error_msg = self.lib.ffi_error_message(result).decode()
            raise RuntimeError(f"Failed to create HTX session: {error_msg}")

        return handle.value

    def send_htx_data(self, handle: int, stream_id: int, data: bytes) -> bytes:
        """Send data on HTX stream"""

        # Prepare input data
        data_array = (c_ubyte * len(data))(*data)
        output_buffer = FFIBuffer()

        # Call FFI function
        result = self.lib.htx_session_send_data(
            handle,
            stream_id,
            data_array,
            len(data),
            ctypes.byref(output_buffer)
        )

        if result != FFIResult.SUCCESS:
            error_msg = self.lib.ffi_error_message(result).decode()
            raise RuntimeError(f"Failed to send HTX data: {error_msg}")

        # Extract output data
        output_data = output_buffer.to_bytes()

        # Free buffer
        self.lib.ffi_buffer_free(output_buffer)

        return output_data

    def free_htx_session(self, handle: int):
        """Free HTX session"""
        self.lib.htx_session_free(handle)
```

### High-Level Python Wrapper
**File**: `py/aivillage/p2p/betanet/client.py`

```python
from typing import Optional, Dict, Any, Callable
import asyncio
import threading
from .ffi_bindings import BetanetFFI, FFIResult

class HtxSession:
    """High-level Python wrapper for HTX sessions"""

    def __init__(self, config: Dict[str, Any], is_initiator: bool = True):
        self.ffi = BetanetFFI()
        self.handle = self.ffi.create_htx_session(config, is_initiator)
        self.is_closed = False
        self._lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def send_data(self, stream_id: int, data: bytes) -> bytes:
        """Send data on stream and return encrypted packet"""
        with self._lock:
            if self.is_closed:
                raise RuntimeError("Session is closed")

            return self.ffi.send_htx_data(self.handle, stream_id, data)

    async def send_data_async(self, stream_id: int, data: bytes) -> bytes:
        """Async version of send_data"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.send_data, stream_id, data)

    def close(self):
        """Close the session and free resources"""
        with self._lock:
            if not self.is_closed:
                self.ffi.free_htx_session(self.handle)
                self.is_closed = True

class MixnodeClient:
    """High-level Python wrapper for mixnode operations"""

    def __init__(self, config: Dict[str, Any]):
        self.ffi = BetanetFFI()
        self.handle = self.ffi.create_mixnode(config)
        self.is_closed = False

    def process_packet(self, packet: bytes) -> Optional[bytes]:
        """Process packet through mixnode"""
        if len(packet) != 2048:
            raise ValueError("Sphinx packets must be exactly 2048 bytes")

        if self.is_closed:
            raise RuntimeError("Mixnode is closed")

        return self.ffi.process_mixnode_packet(self.handle, packet)

    def close(self):
        """Close mixnode and free resources"""
        if not self.is_closed:
            self.ffi.free_mixnode(self.handle)
            self.is_closed = True

# Example usage
async def example_usage():
    """Example of using the Python wrapper"""

    # Create HTX session
    config = {
        "listen_addr": "127.0.0.1:9000",
        "enable_tcp": True,
        "enable_noise_xk": True,
        "max_connections": 100,
    }

    with HtxSession(config, is_initiator=True) as session:
        # Send data on stream
        message = b"Hello, AIVillage!"
        encrypted_packet = await session.send_data_async(1, message)
        print(f"Sent {len(encrypted_packet)} bytes")

    # Create mixnode
    mixnode_config = {
        "node_id": "mix-001",
        "listen_addr": "127.0.0.1:9001",
        "mix_strategy": 0,  # threshold
        "mix_threshold": 5,
        "min_delay_ms": 10,
        "max_delay_ms": 1000,
    }

    mixnode = MixnodeClient(mixnode_config)
    try:
        # Process dummy packet
        dummy_packet = b"\x00" * 2048
        result = mixnode.process_packet(dummy_packet)
        if result:
            print(f"Forwarded packet: {len(result)} bytes")
        else:
            print("Packet delivered to destination")
    finally:
        mixnode.close()

if __name__ == "__main__":
    asyncio.run(example_usage())
```

## Android JNI Integration

### JNI C Bridge
**File**: `platforms/android/jni/betanet_jni.c`

```c
#include <jni.h>
#include <string.h>
#include <stdlib.h>
#include "betanet_ffi.h"

// JNI exception handling
void throw_java_exception(JNIEnv *env, const char *exception_class, const char *message) {
    jclass exception = (*env)->FindClass(env, exception_class);
    if (exception != NULL) {
        (*env)->ThrowNew(env, exception, message);
    }
}

// Convert Java byte array to C buffer
uint8_t* java_byte_array_to_c(JNIEnv *env, jbyteArray array, size_t *length) {
    if (array == NULL) {
        *length = 0;
        return NULL;
    }

    *length = (*env)->GetArrayLength(env, array);
    uint8_t *buffer = malloc(*length);
    if (buffer == NULL) {
        throw_java_exception(env, "java/lang/OutOfMemoryError", "Failed to allocate buffer");
        return NULL;
    }

    (*env)->GetByteArrayRegion(env, array, 0, *length, (jbyte*)buffer);
    return buffer;
}

// Convert C buffer to Java byte array
jbyteArray c_buffer_to_java_byte_array(JNIEnv *env, const uint8_t *buffer, size_t length) {
    if (buffer == NULL || length == 0) {
        return NULL;
    }

    jbyteArray result = (*env)->NewByteArray(env, length);
    if (result == NULL) {
        throw_java_exception(env, "java/lang/OutOfMemoryError", "Failed to create byte array");
        return NULL;
    }

    (*env)->SetByteArrayRegion(env, result, 0, length, (const jbyte*)buffer);
    return result;
}

// HTX Session JNI functions
JNIEXPORT jlong JNICALL
Java_com_aivillage_betanet_HtxSession_nativeCreateSession(
    JNIEnv *env,
    jobject thiz,
    jstring listen_addr,
    jboolean enable_tcp,
    jboolean enable_quic,
    jboolean enable_noise_xk,
    jint max_connections,
    jlong connection_timeout_secs,
    jbyteArray static_private_key,
    jboolean is_initiator
) {
    // Convert Java string to C string
    const char *c_listen_addr = (*env)->GetStringUTFChars(env, listen_addr, 0);

    // Convert private key
    size_t key_len = 0;
    uint8_t *key_data = java_byte_array_to_c(env, static_private_key, &key_len);

    // Create configuration
    HtxConfig config = {
        .listen_addr = c_listen_addr,
        .enable_tcp = enable_tcp,
        .enable_quic = enable_quic,
        .enable_noise_xk = enable_noise_xk,
        .max_connections = max_connections,
        .connection_timeout_secs = connection_timeout_secs,
        .static_private_key = key_data,
        .static_private_key_len = key_len,
    };

    // Create session
    HtxSessionHandle *handle = NULL;
    FFIResult result = htx_session_new(&config, is_initiator, &handle);

    // Cleanup
    (*env)->ReleaseStringUTFChars(env, listen_addr, c_listen_addr);
    if (key_data) free(key_data);

    if (result != FFI_SUCCESS) {
        const char *error_msg = ffi_error_message(result);
        throw_java_exception(env, "com/aivillage/betanet/BetanetException", error_msg);
        return 0;
    }

    return (jlong)handle;
}

JNIEXPORT jbyteArray JNICALL
Java_com_aivillage_betanet_HtxSession_nativeSendData(
    JNIEnv *env,
    jobject thiz,
    jlong handle,
    jint stream_id,
    jbyteArray data
) {
    if (handle == 0) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", "Invalid session handle");
        return NULL;
    }

    // Convert input data
    size_t data_len = 0;
    uint8_t *input_data = java_byte_array_to_c(env, data, &data_len);
    if (input_data == NULL && data_len > 0) {
        return NULL; // Exception already thrown
    }

    // Send data
    FFIBuffer output_buffer;
    FFIResult result = htx_session_send_data(
        (HtxSessionHandle*)handle,
        stream_id,
        input_data,
        data_len,
        &output_buffer
    );

    // Cleanup input
    if (input_data) free(input_data);

    if (result != FFI_SUCCESS) {
        const char *error_msg = ffi_error_message(result);
        throw_java_exception(env, "com/aivillage/betanet/BetanetException", error_msg);
        return NULL;
    }

    // Convert output to Java byte array
    jbyteArray output = c_buffer_to_java_byte_array(env, output_buffer.data, output_buffer.length);

    // Free FFI buffer
    ffi_buffer_free(output_buffer);

    return output;
}

JNIEXPORT void JNICALL
Java_com_aivillage_betanet_HtxSession_nativeFreeSession(JNIEnv *env, jobject thiz, jlong handle) {
    if (handle != 0) {
        htx_session_free((HtxSessionHandle*)handle);
    }
}
```

### Android Java Wrapper
**File**: `platforms/android/java/com/aivillage/betanet/HtxSession.java`

```java
package com.aivillage.betanet;

import java.io.Closeable;
import java.nio.ByteBuffer;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class HtxSession implements Closeable {

    static {
        System.loadLibrary("betanet_jni");
    }

    private long nativeHandle;
    private boolean isClosed = false;
    private final Object lock = new Object();
    private final Executor executor = Executors.newCachedThreadPool();

    public static class Config {
        public String listenAddr = "127.0.0.1:9000";
        public boolean enableTcp = true;
        public boolean enableQuic = false;
        public boolean enableNoiseXk = true;
        public int maxConnections = 1000;
        public long connectionTimeoutSecs = 30;
        public byte[] staticPrivateKey = null;

        public Config() {}

        public Config setListenAddr(String addr) {
            this.listenAddr = addr;
            return this;
        }

        public Config setStaticPrivateKey(byte[] key) {
            if (key != null && key.length != 32) {
                throw new IllegalArgumentException("Private key must be 32 bytes");
            }
            this.staticPrivateKey = key;
            return this;
        }
    }

    public HtxSession(Config config, boolean isInitiator) throws BetanetException {
        synchronized (lock) {
            this.nativeHandle = nativeCreateSession(
                config.listenAddr,
                config.enableTcp,
                config.enableQuic,
                config.enableNoiseXk,
                config.maxConnections,
                config.connectionTimeoutSecs,
                config.staticPrivateKey,
                isInitiator
            );

            if (this.nativeHandle == 0) {
                throw new BetanetException("Failed to create HTX session");
            }
        }
    }

    public byte[] sendData(int streamId, byte[] data) throws BetanetException {
        synchronized (lock) {
            if (isClosed) {
                throw new IllegalStateException("Session is closed");
            }

            return nativeSendData(nativeHandle, streamId, data);
        }
    }

    public CompletableFuture<byte[]> sendDataAsync(int streamId, byte[] data) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return sendData(streamId, data);
            } catch (BetanetException e) {
                throw new RuntimeException(e);
            }
        }, executor);
    }

    @Override
    public void close() {
        synchronized (lock) {
            if (!isClosed) {
                nativeFreeSession(nativeHandle);
                isClosed = true;
            }
        }
    }

    // Native method declarations
    private static native long nativeCreateSession(
        String listenAddr,
        boolean enableTcp,
        boolean enableQuic,
        boolean enableNoiseXk,
        int maxConnections,
        long connectionTimeoutSecs,
        byte[] staticPrivateKey,
        boolean isInitiator
    );

    private static native byte[] nativeSendData(long handle, int streamId, byte[] data);
    private static native void nativeFreeSession(long handle);
}
```

## iOS Swift Integration

### Swift C Interop
**File**: `platforms/ios/Sources/BetanetSwift/HtxSession.swift`

```swift
import Foundation

public class HtxSession {

    private var handle: UnsafeMutableRawPointer?
    private let queue = DispatchQueue(label: "com.aivillage.betanet.htx", qos: .userInitiated)
    private var isClosed = false

    public struct Config {
        public var listenAddr: String = "127.0.0.1:9000"
        public var enableTcp: Bool = true
        public var enableQuic: Bool = false
        public var enableNoiseXk: Bool = true
        public var maxConnections: UInt32 = 1000
        public var connectionTimeoutSecs: UInt64 = 30
        public var staticPrivateKey: Data?

        public init() {}
    }

    public enum BetanetError: Error {
        case invalidHandle
        case invalidInput
        case networkError
        case cryptoError
        case outOfMemory
        case timeout
        case permissionDenied
        case unknown(Int32)

        static func from(result: Int32) -> BetanetError? {
            switch result {
            case 0: return nil // Success
            case -1: return .invalidHandle
            case -2: return .invalidInput
            case -3: return .networkError
            case -4: return .cryptoError
            case -5: return .outOfMemory
            case -6: return .timeout
            case -7: return .permissionDenied
            default: return .unknown(result)
            }
        }
    }

    public init(config: Config, isInitiator: Bool = true) throws {
        // Prepare C configuration
        let cConfig = config.listenAddr.withCString { listenAddrPtr in
            return HtxConfig(
                listen_addr: listenAddrPtr,
                enable_tcp: config.enableTcp,
                enable_quic: config.enableQuic,
                enable_noise_xk: config.enableNoiseXk,
                max_connections: config.maxConnections,
                connection_timeout_secs: config.connectionTimeoutSecs,
                static_private_key: config.staticPrivateKey?.withUnsafeBytes { $0.bindMemory(to: UInt8.self).baseAddress },
                static_private_key_len: config.staticPrivateKey?.count ?? 0
            )
        }

        // Create session
        var handlePtr: UnsafeMutableRawPointer?
        let result = htx_session_new(&cConfig, isInitiator, &handlePtr)

        if let error = BetanetError.from(result: result) {
            throw error
        }

        guard let handle = handlePtr else {
            throw BetanetError.invalidHandle
        }

        self.handle = handle
    }

    public func sendData(streamId: UInt32, data: Data) throws -> Data {
        return try queue.sync {
            guard let handle = self.handle, !isClosed else {
                throw BetanetError.invalidHandle
            }

            return try data.withUnsafeBytes { dataPtr in
                var outputBuffer = FFIBuffer()

                let result = htx_session_send_data(
                    handle,
                    streamId,
                    dataPtr.bindMemory(to: UInt8.self).baseAddress,
                    data.count,
                    &outputBuffer
                )

                if let error = BetanetError.from(result: result) {
                    throw error
                }

                defer {
                    ffi_buffer_free(outputBuffer)
                }

                guard let outputDataPtr = outputBuffer.data, outputBuffer.length > 0 else {
                    return Data()
                }

                return Data(bytes: outputDataPtr, count: outputBuffer.length)
            }
        }
    }

    public func sendDataAsync(streamId: UInt32, data: Data) async throws -> Data {
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    let result = try self.sendData(streamId: streamId, data: data)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    public func close() {
        queue.sync {
            if let handle = self.handle, !isClosed {
                htx_session_free(handle)
                self.handle = nil
                isClosed = true
            }
        }
    }

    deinit {
        close()
    }
}

// C function declarations
@_silgen_name("htx_session_new")
func htx_session_new(
    _ config: UnsafePointer<HtxConfig>,
    _ isInitiator: Bool,
    _ handle: UnsafeMutablePointer<UnsafeMutableRawPointer?>
) -> Int32

@_silgen_name("htx_session_send_data")
func htx_session_send_data(
    _ handle: UnsafeMutableRawPointer,
    _ streamId: UInt32,
    _ data: UnsafePointer<UInt8>?,
    _ dataLen: Int,
    _ outBuffer: UnsafeMutablePointer<FFIBuffer>
) -> Int32

@_silgen_name("htx_session_free")
func htx_session_free(_ handle: UnsafeMutableRawPointer)

@_silgen_name("ffi_buffer_free")
func ffi_buffer_free(_ buffer: FFIBuffer)
```

## Build Integration

### Cargo.toml Configuration
**File**: `crates/betanet-ffi/Cargo.toml`

```toml
[package]
name = "betanet-ffi"
version = "0.1.0"
edition = "2021"

[lib]
name = "betanet_ffi"
crate-type = ["cdylib", "staticlib"]

[dependencies]
betanet-htx = { path = "../betanet-htx" }
betanet-mixnode = { path = "../betanet-mixnode" }
agent-fabric = { path = "../agent-fabric" }
twin-vault = { path = "../twin-vault" }

tokio = { version = "1.40", features = ["rt", "rt-multi-thread"] }
bytes = "1.7"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[build-dependencies]
cbindgen = "0.24"

[features]
default = ["all-protocols"]
all-protocols = ["htx", "mixnode", "agent-fabric", "twin-vault"]
htx = []
mixnode = []
agent-fabric = []
twin-vault = []
```

### Build Script
**File**: `crates/betanet-ffi/build.rs`

```rust
extern crate cbindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_file = PathBuf::from(&crate_dir)
        .join("include")
        .join("betanet_ffi.h");

    // Ensure include directory exists
    std::fs::create_dir_all(output_file.parent().unwrap()).unwrap();

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_language(cbindgen::Language::C)
        .with_no_includes()
        .with_sys_include("stdint.h")
        .with_sys_include("stdbool.h")
        .with_pragma_once(true)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(&output_file);

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=build.rs");
}
```

The FFI integration provides comprehensive language bindings while maintaining the security and performance characteristics of the underlying Rust implementation. This enables AIVillage's Rust Client Infrastructure to be seamlessly integrated across diverse platform ecosystems.
