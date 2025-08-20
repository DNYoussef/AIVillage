//! HTX protocol FFI bindings

use crate::common::{BetanetResult, BetanetHandle, BetanetBuffer};
use std::os::raw::{c_char, c_int, c_uint};
use std::ffi::{CStr, CString};
use std::ptr;
use betanet_htx::{Frame, FrameType, HTXClient, HTXServer};

/// HTX frame handle
pub type HTXFrame = BetanetHandle;

/// HTX client handle
pub type HTXClient = BetanetHandle;

/// HTX server handle
pub type HTXServer = BetanetHandle;

/// Create a new HTX frame
///
/// # Arguments
/// * `stream_id` - Stream identifier
/// * `frame_type` - Frame type (DATA=0, WINDOW_UPDATE=1, PING=2, etc.)
/// * `payload` - Frame payload data
///
/// # Returns
/// * Frame handle on success, null on failure
#[no_mangle]
pub extern "C" fn htx_frame_create(
    stream_id: c_uint,
    frame_type: c_uint,
    payload: BetanetBuffer,
) -> *mut HTXFrame {
    let payload_vec = unsafe {
        if payload.data.is_null() {
            Vec::new()
        } else {
            payload.as_slice().to_vec()
        }
    };

    let frame_type = match frame_type {
        0 => FrameType::Data,
        1 => FrameType::WindowUpdate,
        2 => FrameType::Ping,
        3 => FrameType::Reset,
        _ => return ptr::null_mut(),
    };

    let frame = match Frame::new(frame_type, stream_id, payload_vec) {
        Ok(f) => f,
        Err(_) => return ptr::null_mut(),
    };

    Box::into_raw(Box::new(frame)) as *mut HTXFrame
}

/// Encode HTX frame to bytes
///
/// # Arguments
/// * `frame` - Frame handle
/// * `buffer` - Output buffer (will be allocated)
///
/// # Returns
/// * Result code
#[no_mangle]
pub extern "C" fn htx_frame_encode(
    frame: *const HTXFrame,
    buffer: *mut BetanetBuffer,
) -> BetanetResult {
    if frame.is_null() || buffer.is_null() {
        return BetanetResult::InvalidArgument;
    }

    let frame_ref = unsafe { &*(frame as *const Frame) };

    match frame_ref.encode() {
        Ok(data) => {
            unsafe {
                *buffer = BetanetBuffer::from_vec(data);
            }
            BetanetResult::Success
        }
        Err(_) => BetanetResult::InternalError,
    }
}

/// Decode HTX frame from bytes
///
/// # Arguments
/// * `data` - Input buffer containing frame data
///
/// # Returns
/// * Frame handle on success, null on failure
#[no_mangle]
pub extern "C" fn htx_frame_decode(data: BetanetBuffer) -> *mut HTXFrame {
    let data_slice = unsafe {
        if data.data.is_null() {
            return ptr::null_mut();
        }
        data.as_slice()
    };

    let frame = match Frame::decode(data_slice) {
        Ok(f) => f,
        Err(_) => return ptr::null_mut(),
    };

    Box::into_raw(Box::new(frame)) as *mut HTXFrame
}

/// Get frame stream ID
///
/// # Arguments
/// * `frame` - Frame handle
///
/// # Returns
/// * Stream ID, or 0 if frame is invalid
#[no_mangle]
pub extern "C" fn htx_frame_stream_id(frame: *const HTXFrame) -> c_uint {
    if frame.is_null() {
        return 0;
    }

    let frame_ref = unsafe { &*(frame as *const Frame) };
    frame_ref.stream_id()
}

/// Get frame type
///
/// # Arguments
/// * `frame` - Frame handle
///
/// # Returns
/// * Frame type code, or 255 if frame is invalid
#[no_mangle]
pub extern "C" fn htx_frame_type(frame: *const HTXFrame) -> c_uint {
    if frame.is_null() {
        return 255;
    }

    let frame_ref = unsafe { &*(frame as *const Frame) };
    match frame_ref.frame_type() {
        FrameType::Data => 0,
        FrameType::WindowUpdate => 1,
        FrameType::Ping => 2,
        FrameType::Reset => 3,
        _ => 255,
    }
}

/// Get frame payload
///
/// # Arguments
/// * `frame` - Frame handle
/// * `buffer` - Output buffer (will reference frame data)
///
/// # Returns
/// * Result code
#[no_mangle]
pub extern "C" fn htx_frame_payload(
    frame: *const HTXFrame,
    buffer: *mut BetanetBuffer,
) -> BetanetResult {
    if frame.is_null() || buffer.is_null() {
        return BetanetResult::InvalidArgument;
    }

    let frame_ref = unsafe { &*(frame as *const Frame) };
    let payload = frame_ref.payload();

    unsafe {
        *buffer = BetanetBuffer::from_slice(payload);
    }

    BetanetResult::Success
}

/// Free HTX frame
///
/// # Arguments
/// * `frame` - Frame handle to free
#[no_mangle]
pub extern "C" fn htx_frame_free(frame: *mut HTXFrame) {
    if !frame.is_null() {
        unsafe {
            let _ = Box::from_raw(frame as *mut Frame);
        }
    }
}

/// Create HTX client
///
/// # Returns
/// * Client handle on success, null on failure
#[no_mangle]
pub extern "C" fn htx_client_create() -> *mut HTXClient {
    let client = match betanet_htx::HTXClient::new() {
        Ok(c) => c,
        Err(_) => return ptr::null_mut(),
    };

    Box::into_raw(Box::new(client)) as *mut HTXClient
}

/// Connect HTX client to server
///
/// # Arguments
/// * `client` - Client handle
/// * `address` - Server address (null-terminated string)
/// * `port` - Server port
///
/// # Returns
/// * Result code
#[no_mangle]
pub extern "C" fn htx_client_connect(
    client: *mut HTXClient,
    address: *const c_char,
    port: c_uint,
) -> BetanetResult {
    if client.is_null() || address.is_null() {
        return BetanetResult::InvalidArgument;
    }

    let client_ref = unsafe { &mut *(client as *mut betanet_htx::HTXClient) };

    let address_str = unsafe {
        match CStr::from_ptr(address).to_str() {
            Ok(s) => s,
            Err(_) => return BetanetResult::InvalidArgument,
        }
    };

    // Note: This would need actual async runtime setup in a real implementation
    match client_ref.connect(address_str, port as u16) {
        Ok(_) => BetanetResult::Success,
        Err(_) => BetanetResult::NetworkError,
    }
}

/// Send frame via HTX client
///
/// # Arguments
/// * `client` - Client handle
/// * `frame` - Frame to send
///
/// # Returns
/// * Result code
#[no_mangle]
pub extern "C" fn htx_client_send_frame(
    client: *mut HTXClient,
    frame: *const HTXFrame,
) -> BetanetResult {
    if client.is_null() || frame.is_null() {
        return BetanetResult::InvalidArgument;
    }

    let client_ref = unsafe { &mut *(client as *mut betanet_htx::HTXClient) };
    let frame_ref = unsafe { &*(frame as *const Frame) };

    match client_ref.send_frame(frame_ref.clone()) {
        Ok(_) => BetanetResult::Success,
        Err(_) => BetanetResult::NetworkError,
    }
}

/// Free HTX client
///
/// # Arguments
/// * `client` - Client handle to free
#[no_mangle]
pub extern "C" fn htx_client_free(client: *mut HTXClient) {
    if !client.is_null() {
        unsafe {
            let _ = Box::from_raw(client as *mut betanet_htx::HTXClient);
        }
    }
}
