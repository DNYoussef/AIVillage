//! Betanet C FFI Library
//!
//! Provides C-compatible FFI bindings for Betanet components with safe APIs,
//! async runtime management, and comprehensive error handling.
//!
//! # Memory Management
//! 
//! - All pointers returned by `*_create()` functions must be freed with corresponding `*_destroy()` functions
//! - String parameters passed to functions must be null-terminated and valid for the duration of the call
//! - Callback functions may be called from any thread and must be thread-safe
//! - User data pointers are passed through unchanged and remain owned by the caller
//!
//! # Thread Safety
//!
//! - All functions are thread-safe unless otherwise noted
//! - Callbacks may be invoked from background threads
//! - The library uses internal async runtime for non-blocking operations
//!
//! # Error Handling
//!
//! - Functions return BetanetResult enum values indicating success or failure
//! - Additional error information available via betanet_get_last_error()
//! - Error strings are valid until the next function call or betanet_clear_error()

#![deny(warnings)]
#![deny(clippy::all)]
#![allow(clippy::missing_safety_doc)]

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ptr;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use tokio::runtime::Runtime;
use tokio::sync::mpsc;

/// Result codes for C API
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BetanetResult {
    Success = 0,
    Error = 1,
    InvalidParameter = 2,
    NetworkError = 3,
    CryptoError = 4,
    Timeout = 5,
    NotConnected = 6,
    AlreadyConnected = 7,
    BufferTooSmall = 8,
}

/// Connection state
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BetanetConnectionState {
    Disconnected = 0,
    Connecting = 1,
    Connected = 2,
    Disconnecting = 3,
    Error = 4,
}

/// Transport type
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum BetanetTransport {
    Tcp = 0,
    Quic = 1,
    NoiseXk = 2,
    HybridKem = 3,
}

/// Opaque handle for HTX client
pub struct BetanetHtxClient {
    runtime: Arc<Runtime>,
    state: Arc<Mutex<ClientState>>,
    rx: Arc<Mutex<mpsc::Receiver<Vec<u8>>>>,
    tx: mpsc::Sender<Vec<u8>>,
}

/// Opaque handle for HTX server
pub struct BetanetHtxServer {
    runtime: Arc<Runtime>,
    state: Arc<Mutex<ServerState>>,
    connections: Arc<Mutex<HashMap<u32, Connection>>>,
}

/// Opaque handle for mixnode
#[allow(dead_code)]
pub struct BetanetMixnode {
    runtime: Arc<Runtime>,
    running: Arc<AtomicBool>,
}

#[allow(dead_code)]
struct ClientState {
    connection_state: BetanetConnectionState,
    remote_addr: Option<String>,
    error_message: Option<String>,
}

#[allow(dead_code)]
struct ServerState {
    listen_addr: String,
    running: bool,
    error_message: Option<String>,
}

#[allow(dead_code)]
struct Connection {
    id: u32,
    remote_addr: String,
    tx: mpsc::Sender<Vec<u8>>,
    rx: mpsc::Receiver<Vec<u8>>,
}

/// C-compatible configuration structure
#[repr(C)]
pub struct BetanetConfig {
    /// Listen address (null-terminated string)
    pub listen_addr: *const c_char,
    /// Server name for TLS (null-terminated string, optional)
    pub server_name: *const c_char,
    /// Transport type
    pub transport: BetanetTransport,
    /// Maximum connections
    pub max_connections: c_uint,
    /// Connection timeout in seconds
    pub connection_timeout_secs: c_uint,
    /// Keep-alive interval in seconds
    pub keepalive_interval_secs: c_uint,
    /// Enable compression
    pub enable_compression: c_int,
}

/// Callback function types
pub type BetanetDataCallback = extern "C" fn(user_data: *mut c_void, data: *const u8, len: c_uint);

pub type BetanetConnectionCallback =
    extern "C" fn(user_data: *mut c_void, state: BetanetConnectionState);

pub type BetanetErrorCallback =
    extern "C" fn(user_data: *mut c_void, error_code: BetanetResult, error_msg: *const c_char);

// Thread-local error storage
thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<CString>> = std::cell::RefCell::new(None);
}

fn set_last_error(msg: String) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(msg).ok();
    });
}

/// Initialize the Betanet library
#[no_mangle]
pub extern "C" fn betanet_init() -> BetanetResult {
    // Initialize logging if not already initialized
    let _ = tracing_subscriber::fmt::try_init();
    BetanetResult::Success
}

/// Create HTX client with configuration
#[no_mangle]
pub extern "C" fn betanet_htx_client_create(config: *const BetanetConfig) -> *mut BetanetHtxClient {
    if config.is_null() {
        set_last_error("Configuration is null".to_string());
        return ptr::null_mut();
    }

    let config = unsafe { &*config };

    // Parse listen address
    let _listen_addr = if config.listen_addr.is_null() {
        "127.0.0.1:9000".to_string()
    } else {
        unsafe {
            match CStr::from_ptr(config.listen_addr).to_str() {
                Ok(s) => s.to_string(),
                Err(_) => {
                    set_last_error("Invalid UTF-8 in listen address".to_string());
                    return ptr::null_mut();
                }
            }
        }
    };

    // Create async runtime
    let runtime = match Runtime::new() {
        Ok(rt) => Arc::new(rt),
        Err(e) => {
            set_last_error(format!("Failed to create runtime: {}", e));
            return ptr::null_mut();
        }
    };

    // Create channels for communication
    let (tx, rx) = mpsc::channel(1024);

    let client = BetanetHtxClient {
        runtime,
        state: Arc::new(Mutex::new(ClientState {
            connection_state: BetanetConnectionState::Disconnected,
            remote_addr: None,
            error_message: None,
        })),
        rx: Arc::new(Mutex::new(rx)),
        tx,
    };

    Box::into_raw(Box::new(client))
}

/// Connect HTX client to server (async)
#[no_mangle]
pub extern "C" fn betanet_htx_client_connect_async(
    client: *mut BetanetHtxClient,
    addr: *const c_char,
    callback: BetanetConnectionCallback,
    user_data: *mut c_void,
) -> BetanetResult {
    if client.is_null() || addr.is_null() {
        set_last_error("Invalid parameters".to_string());
        return BetanetResult::InvalidParameter;
    }

    let client = unsafe { &mut *client };

    let addr_str = unsafe {
        match CStr::from_ptr(addr).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => {
                set_last_error("Invalid UTF-8 in address".to_string());
                return BetanetResult::InvalidParameter;
            }
        }
    };

    // Update state
    {
        let mut state = client.state.lock().unwrap();
        if state.connection_state == BetanetConnectionState::Connected {
            return BetanetResult::AlreadyConnected;
        }
        state.connection_state = BetanetConnectionState::Connecting;
        state.remote_addr = Some(addr_str.clone());
    }

    // Spawn async connection task
    // Note: user_data is a raw pointer and not Send, so we convert to usize
    let user_data_usize = user_data as usize;
    let state = client.state.clone();
    client.runtime.spawn(async move {
        // Simulate connection (would use actual HTX client here)
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Update state and call callback
        {
            let mut state = state.lock().unwrap();
            state.connection_state = BetanetConnectionState::Connected;
        }

        // Convert back to pointer for callback
        let user_data_ptr = user_data_usize as *mut c_void;
        callback(user_data_ptr, BetanetConnectionState::Connected);
    });

    BetanetResult::Success
}

/// Send data via HTX client (async)
#[no_mangle]
pub extern "C" fn betanet_htx_client_send_async(
    client: *mut BetanetHtxClient,
    data: *const u8,
    len: c_uint,
    callback: BetanetErrorCallback,
    user_data: *mut c_void,
) -> BetanetResult {
    if client.is_null() || data.is_null() || len == 0 {
        set_last_error("Invalid parameters".to_string());
        return BetanetResult::InvalidParameter;
    }

    let client = unsafe { &mut *client };

    // Check connection state
    {
        let state = client.state.lock().unwrap();
        if state.connection_state != BetanetConnectionState::Connected {
            return BetanetResult::NotConnected;
        }
    }

    // Copy data
    let data_vec = unsafe { std::slice::from_raw_parts(data, len as usize) }.to_vec();

    // Send data async
    // Note: user_data is a raw pointer and not Send, so we convert to usize
    let user_data_usize = user_data as usize;
    let tx = client.tx.clone();
    client.runtime.spawn(async move {
        match tx.send(data_vec).await {
            Ok(_) => {
                // Convert back to pointer for callback after await
                let user_data_ptr = user_data_usize as *mut c_void;
                callback(user_data_ptr, BetanetResult::Success, ptr::null());
            }
            Err(e) => {
                // Convert back to pointer for callback after await
                let user_data_ptr = user_data_usize as *mut c_void;
                let error_msg = CString::new(format!("Send failed: {}", e)).unwrap();
                callback(
                    user_data_ptr,
                    BetanetResult::NetworkError,
                    error_msg.as_ptr(),
                );
            }
        }
    });

    BetanetResult::Success
}

/// Receive data from HTX client (non-blocking)
#[no_mangle]
pub extern "C" fn betanet_htx_client_recv(
    client: *mut BetanetHtxClient,
    buffer: *mut u8,
    buffer_size: c_uint,
    received_len: *mut c_uint,
) -> BetanetResult {
    if client.is_null() || buffer.is_null() || buffer_size == 0 || received_len.is_null() {
        set_last_error("Invalid parameters".to_string());
        return BetanetResult::InvalidParameter;
    }

    let client = unsafe { &mut *client };

    // Try to receive data (non-blocking)
    let mut rx = match client.rx.try_lock() {
        Ok(rx) => rx,
        Err(_) => return BetanetResult::Error,
    };

    match rx.try_recv() {
        Ok(data) => {
            let copy_len = std::cmp::min(data.len(), buffer_size as usize);
            unsafe {
                ptr::copy_nonoverlapping(data.as_ptr(), buffer, copy_len);
                *received_len = copy_len as c_uint;
            }

            if data.len() > buffer_size as usize {
                BetanetResult::BufferTooSmall
            } else {
                BetanetResult::Success
            }
        }
        Err(mpsc::error::TryRecvError::Empty) => {
            unsafe {
                *received_len = 0;
            }
            BetanetResult::Success
        }
        Err(mpsc::error::TryRecvError::Disconnected) => BetanetResult::NotConnected,
    }
}

/// Create HTX server
#[no_mangle]
pub extern "C" fn betanet_htx_server_create(config: *const BetanetConfig) -> *mut BetanetHtxServer {
    if config.is_null() {
        set_last_error("Configuration is null".to_string());
        return ptr::null_mut();
    }

    let config = unsafe { &*config };

    let listen_addr = if config.listen_addr.is_null() {
        "127.0.0.1:9000".to_string()
    } else {
        unsafe {
            match CStr::from_ptr(config.listen_addr).to_str() {
                Ok(s) => s.to_string(),
                Err(_) => {
                    set_last_error("Invalid UTF-8 in listen address".to_string());
                    return ptr::null_mut();
                }
            }
        }
    };

    let runtime = match Runtime::new() {
        Ok(rt) => Arc::new(rt),
        Err(e) => {
            set_last_error(format!("Failed to create runtime: {}", e));
            return ptr::null_mut();
        }
    };

    let server = BetanetHtxServer {
        runtime,
        state: Arc::new(Mutex::new(ServerState {
            listen_addr,
            running: false,
            error_message: None,
        })),
        connections: Arc::new(Mutex::new(HashMap::new())),
    };

    Box::into_raw(Box::new(server))
}

/// Start HTX server (async)
#[no_mangle]
pub extern "C" fn betanet_htx_server_start_async(
    server: *mut BetanetHtxServer,
    callback: BetanetConnectionCallback,
    user_data: *mut c_void,
) -> BetanetResult {
    if server.is_null() {
        set_last_error("Server is null".to_string());
        return BetanetResult::InvalidParameter;
    }

    let server = unsafe { &mut *server };

    // Update state
    {
        let mut state = server.state.lock().unwrap();
        if state.running {
            return BetanetResult::AlreadyConnected;
        }
        state.running = true;
    }

    // Spawn async server task
    // Note: user_data is a raw pointer and not Send, so we convert to usize
    let user_data_usize = user_data as usize;
    let _state = server.state.clone();
    server.runtime.spawn(async move {
        // Simulate server startup
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        // Convert back to pointer for callback
        let user_data_ptr = user_data_usize as *mut c_void;
        callback(user_data_ptr, BetanetConnectionState::Connected);
    });

    BetanetResult::Success
}

/// Accept connection on server
#[no_mangle]
pub extern "C" fn betanet_htx_server_accept(
    server: *mut BetanetHtxServer,
    connection_id: *mut c_uint,
) -> BetanetResult {
    if server.is_null() || connection_id.is_null() {
        set_last_error("Invalid parameters".to_string());
        return BetanetResult::InvalidParameter;
    }

    let server = unsafe { &mut *server };

    // Check if server is running
    {
        let state = server.state.lock().unwrap();
        if !state.running {
            return BetanetResult::NotConnected;
        }
    }

    // Simulate accepting connection
    let (tx, rx) = mpsc::channel(1024);
    // Simple connection ID generation (in production, use proper random or counter)
    let conn_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as u32;

    let connection = Connection {
        id: conn_id,
        remote_addr: "127.0.0.1:12345".to_string(),
        tx,
        rx,
    };

    {
        let mut connections = server.connections.lock().unwrap();
        connections.insert(conn_id, connection);
    }

    unsafe {
        *connection_id = conn_id;
    }
    BetanetResult::Success
}

/// Clean up HTX client
#[no_mangle]
pub extern "C" fn betanet_htx_client_destroy(client: *mut BetanetHtxClient) {
    if !client.is_null() {
        unsafe {
            let _ = Box::from_raw(client);
        }
    }
}

/// Clean up HTX server
#[no_mangle]
pub extern "C" fn betanet_htx_server_destroy(server: *mut BetanetHtxServer) {
    if !server.is_null() {
        unsafe {
            let _ = Box::from_raw(server);
        }
    }
}

/// Get library version
#[no_mangle]
pub extern "C" fn betanet_get_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Get last error message
#[no_mangle]
pub extern "C" fn betanet_get_last_error() -> *const c_char {
    LAST_ERROR.with(|e| match &*e.borrow() {
        Some(err) => err.as_ptr(),
        None => ptr::null(),
    })
}

/// Clear last error
#[no_mangle]
pub extern "C" fn betanet_clear_error() {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = None;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_version() {
        let version = unsafe { CStr::from_ptr(betanet_get_version()) };
        assert!(!version.to_string_lossy().is_empty());
    }

    #[test]
    fn test_client_creation() {
        let addr = CString::new("127.0.0.1:9001").unwrap();
        let config = BetanetConfig {
            listen_addr: addr.as_ptr(),
            server_name: ptr::null(),
            transport: BetanetTransport::Tcp,
            max_connections: 100,
            connection_timeout_secs: 30,
            keepalive_interval_secs: 10,
            enable_compression: 0,
        };

        let client = betanet_htx_client_create(&config);
        assert!(!client.is_null());

        betanet_htx_client_destroy(client);
    }

    #[test]
    fn test_server_creation() {
        let addr = CString::new("0.0.0.0:9002").unwrap();
        let config = BetanetConfig {
            listen_addr: addr.as_ptr(),
            server_name: ptr::null(),
            transport: BetanetTransport::NoiseXk,
            max_connections: 50,
            connection_timeout_secs: 60,
            keepalive_interval_secs: 15,
            enable_compression: 1,
        };

        let server = betanet_htx_server_create(&config);
        assert!(!server.is_null());

        betanet_htx_server_destroy(server);
    }

    #[test]
    fn test_error_handling() {
        betanet_clear_error();
        let error = betanet_get_last_error();
        assert!(error.is_null());

        // Trigger an error
        let client = betanet_htx_client_create(ptr::null());
        assert!(client.is_null());

        let error = betanet_get_last_error();
        assert!(!error.is_null());

        let error_msg = unsafe { CStr::from_ptr(error) };
        assert!(error_msg.to_string_lossy().contains("Configuration"));
    }
}
