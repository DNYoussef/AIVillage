//! Mixnode FFI bindings

use crate::common::{BetanetResult, BetanetHandle, BetanetBuffer};
use std::os::raw::{c_char, c_int, c_uint, c_double};
use std::ffi::{CStr, CString};
use std::ptr;
use betanet_mixnode::{Mixnode, MixnodeConfig, SphinxPacket};

/// Mixnode handle
pub type MixnodeHandle = BetanetHandle;

/// Sphinx packet handle
pub type SphinxPacketHandle = BetanetHandle;

/// Mixnode configuration structure
#[repr(C)]
pub struct MixnodeConfigFFI {
    /// Maximum packets per second
    pub max_pps: c_uint,
    /// Delay pool size
    pub delay_pool_size: c_uint,
    /// Cover traffic rate (packets per second)
    pub cover_traffic_rate: c_double,
    /// Enable VRF delay
    pub enable_vrf_delay: c_int,
}

/// Create mixnode with configuration
///
/// # Arguments
/// * `config` - Mixnode configuration
///
/// # Returns
/// * Mixnode handle on success, null on failure
#[no_mangle]
pub extern "C" fn mixnode_create(config: *const MixnodeConfigFFI) -> *mut MixnodeHandle {
    let config_data = if config.is_null() {
        MixnodeConfig::default()
    } else {
        let config_ref = unsafe { &*config };
        MixnodeConfig {
            max_packets_per_second: config_ref.max_pps,
            delay_pool_size: config_ref.delay_pool_size as usize,
            cover_traffic_rate: config_ref.cover_traffic_rate,
            enable_vrf_delay: config_ref.enable_vrf_delay != 0,
            ..Default::default()
        }
    };

    let mixnode = match Mixnode::new(config_data) {
        Ok(m) => m,
        Err(_) => return ptr::null_mut(),
    };

    Box::into_raw(Box::new(mixnode)) as *mut MixnodeHandle
}

/// Start mixnode operation
///
/// # Arguments
/// * `mixnode` - Mixnode handle
///
/// # Returns
/// * Result code
#[no_mangle]
pub extern "C" fn mixnode_start(mixnode: *mut MixnodeHandle) -> BetanetResult {
    if mixnode.is_null() {
        return BetanetResult::InvalidArgument;
    }

    let mixnode_ref = unsafe { &mut *(mixnode as *mut Mixnode) };

    match mixnode_ref.start() {
        Ok(_) => BetanetResult::Success,
        Err(_) => BetanetResult::InternalError,
    }
}

/// Stop mixnode operation
///
/// # Arguments
/// * `mixnode` - Mixnode handle
///
/// # Returns
/// * Result code
#[no_mangle]
pub extern "C" fn mixnode_stop(mixnode: *mut MixnodeHandle) -> BetanetResult {
    if mixnode.is_null() {
        return BetanetResult::InvalidArgument;
    }

    let mixnode_ref = unsafe { &mut *(mixnode as *mut Mixnode) };

    match mixnode_ref.stop() {
        Ok(_) => BetanetResult::Success,
        Err(_) => BetanetResult::InternalError,
    }
}

/// Process incoming packet
///
/// # Arguments
/// * `mixnode` - Mixnode handle
/// * `packet_data` - Incoming packet data
/// * `output_packet` - Output packet buffer (will be allocated if packet should be forwarded)
///
/// # Returns
/// * Result code
#[no_mangle]
pub extern "C" fn mixnode_process_packet(
    mixnode: *mut MixnodeHandle,
    packet_data: BetanetBuffer,
    output_packet: *mut BetanetBuffer,
) -> BetanetResult {
    if mixnode.is_null() || output_packet.is_null() {
        return BetanetResult::InvalidArgument;
    }

    let mixnode_ref = unsafe { &mut *(mixnode as *mut Mixnode) };

    let input_data = unsafe {
        if packet_data.data.is_null() {
            return BetanetResult::InvalidArgument;
        }
        packet_data.as_slice()
    };

    match mixnode_ref.process_packet(input_data) {
        Ok(Some(output_data)) => {
            unsafe {
                *output_packet = BetanetBuffer::from_vec(output_data);
            }
            BetanetResult::Success
        }
        Ok(None) => {
            // Packet consumed (final destination or dropped)
            unsafe {
                *output_packet = BetanetBuffer::new();
            }
            BetanetResult::Success
        }
        Err(_) => BetanetResult::InternalError,
    }
}

/// Get mixnode statistics
///
/// # Arguments
/// * `mixnode` - Mixnode handle
/// * `packets_processed` - Output for packets processed count
/// * `packets_forwarded` - Output for packets forwarded count
/// * `packets_dropped` - Output for packets dropped count
/// * `current_pps` - Output for current packets per second
///
/// # Returns
/// * Result code
#[no_mangle]
pub extern "C" fn mixnode_get_stats(
    mixnode: *const MixnodeHandle,
    packets_processed: *mut c_uint,
    packets_forwarded: *mut c_uint,
    packets_dropped: *mut c_uint,
    current_pps: *mut c_double,
) -> BetanetResult {
    if mixnode.is_null() {
        return BetanetResult::InvalidArgument;
    }

    let mixnode_ref = unsafe { &*(mixnode as *const Mixnode) };
    let stats = mixnode_ref.get_stats();

    if !packets_processed.is_null() {
        unsafe { *packets_processed = stats.packets_processed as c_uint; }
    }
    if !packets_forwarded.is_null() {
        unsafe { *packets_forwarded = stats.packets_forwarded as c_uint; }
    }
    if !packets_dropped.is_null() {
        unsafe { *packets_dropped = stats.packets_dropped as c_uint; }
    }
    if !current_pps.is_null() {
        unsafe { *current_pps = stats.current_pps; }
    }

    BetanetResult::Success
}

/// Free mixnode
///
/// # Arguments
/// * `mixnode` - Mixnode handle to free
#[no_mangle]
pub extern "C" fn mixnode_free(mixnode: *mut MixnodeHandle) {
    if !mixnode.is_null() {
        unsafe {
            let _ = Box::from_raw(mixnode as *mut Mixnode);
        }
    }
}

/// Create Sphinx packet
///
/// # Arguments
/// * `payload` - Packet payload
/// * `route` - Routing information (comma-separated node addresses)
///
/// # Returns
/// * Sphinx packet handle on success, null on failure
#[no_mangle]
pub extern "C" fn sphinx_packet_create(
    payload: BetanetBuffer,
    route: *const c_char,
) -> *mut SphinxPacketHandle {
    let payload_data = unsafe {
        if payload.data.is_null() {
            Vec::new()
        } else {
            payload.as_slice().to_vec()
        }
    };

    let route_str = if route.is_null() {
        return ptr::null_mut();
    } else {
        unsafe {
            match CStr::from_ptr(route).to_str() {
                Ok(s) => s,
                Err(_) => return ptr::null_mut(),
            }
        }
    };

    // Parse route string (simplified)
    let route_nodes: Vec<&str> = route_str.split(',').collect();

    let packet = match SphinxPacket::new(payload_data, &route_nodes) {
        Ok(p) => p,
        Err(_) => return ptr::null_mut(),
    };

    Box::into_raw(Box::new(packet)) as *mut SphinxPacketHandle
}

/// Encode Sphinx packet to bytes
///
/// # Arguments
/// * `packet` - Sphinx packet handle
/// * `buffer` - Output buffer (will be allocated)
///
/// # Returns
/// * Result code
#[no_mangle]
pub extern "C" fn sphinx_packet_encode(
    packet: *const SphinxPacketHandle,
    buffer: *mut BetanetBuffer,
) -> BetanetResult {
    if packet.is_null() || buffer.is_null() {
        return BetanetResult::InvalidArgument;
    }

    let packet_ref = unsafe { &*(packet as *const SphinxPacket) };

    match packet_ref.encode() {
        Ok(data) => {
            unsafe {
                *buffer = BetanetBuffer::from_vec(data);
            }
            BetanetResult::Success
        }
        Err(_) => BetanetResult::InternalError,
    }
}

/// Decode Sphinx packet from bytes
///
/// # Arguments
/// * `data` - Input buffer containing packet data
///
/// # Returns
/// * Sphinx packet handle on success, null on failure
#[no_mangle]
pub extern "C" fn sphinx_packet_decode(data: BetanetBuffer) -> *mut SphinxPacketHandle {
    let data_slice = unsafe {
        if data.data.is_null() {
            return ptr::null_mut();
        }
        data.as_slice()
    };

    let packet = match SphinxPacket::decode(data_slice) {
        Ok(p) => p,
        Err(_) => return ptr::null_mut(),
    };

    Box::into_raw(Box::new(packet)) as *mut SphinxPacketHandle
}

/// Free Sphinx packet
///
/// # Arguments
/// * `packet` - Sphinx packet handle to free
#[no_mangle]
pub extern "C" fn sphinx_packet_free(packet: *mut SphinxPacketHandle) {
    if !packet.is_null() {
        unsafe {
            let _ = Box::from_raw(packet as *mut SphinxPacket);
        }
    }
}
