//! uTLS and fingerprinting FFI bindings

use crate::common::{BetanetResult, BetanetHandle, BetanetBuffer};
use std::os::raw::{c_char, c_int, c_uint};
use std::ffi::{CStr, CString};
use std::ptr;
use betanet_utls::{JA3Generator, JA4Generator, ClientHelloTemplate};

/// uTLS template handle
pub type UTLSTemplate = BetanetHandle;

/// JA3 fingerprint generator handle
pub type JA3Generator = BetanetHandle;

/// JA4 fingerprint generator handle
pub type JA4Generator = BetanetHandle;

/// Create JA3 fingerprint generator
///
/// # Returns
/// * Generator handle on success, null on failure
#[no_mangle]
pub extern "C" fn utls_ja3_generator_create() -> *mut JA3Generator {
    let generator = betanet_utls::JA3Generator::new();
    Box::into_raw(Box::new(generator)) as *mut JA3Generator
}

/// Generate JA3 fingerprint from ClientHello
///
/// # Arguments
/// * `generator` - JA3 generator handle
/// * `client_hello` - ClientHello packet data
/// * `fingerprint` - Output buffer for fingerprint (will be allocated)
///
/// # Returns
/// * Result code
#[no_mangle]
pub extern "C" fn utls_ja3_generate(
    generator: *mut JA3Generator,
    client_hello: BetanetBuffer,
    fingerprint: *mut BetanetBuffer,
) -> BetanetResult {
    if generator.is_null() || fingerprint.is_null() {
        return BetanetResult::InvalidArgument;
    }

    let generator_ref = unsafe { &mut *(generator as *mut betanet_utls::JA3Generator) };

    let client_hello_data = unsafe {
        if client_hello.data.is_null() {
            return BetanetResult::InvalidArgument;
        }
        client_hello.as_slice()
    };

    match generator_ref.generate_fingerprint(client_hello_data) {
        Ok(fp) => {
            let fp_bytes = fp.as_bytes().to_vec();
            unsafe {
                *fingerprint = BetanetBuffer::from_vec(fp_bytes);
            }
            BetanetResult::Success
        }
        Err(_) => BetanetResult::ParseError,
    }
}

/// Free JA3 generator
///
/// # Arguments
/// * `generator` - Generator handle to free
#[no_mangle]
pub extern "C" fn utls_ja3_generator_free(generator: *mut JA3Generator) {
    if !generator.is_null() {
        unsafe {
            let _ = Box::from_raw(generator as *mut betanet_utls::JA3Generator);
        }
    }
}

/// Create JA4 fingerprint generator
///
/// # Returns
/// * Generator handle on success, null on failure
#[no_mangle]
pub extern "C" fn utls_ja4_generator_create() -> *mut JA4Generator {
    let generator = betanet_utls::JA4Generator::new();
    Box::into_raw(Box::new(generator)) as *mut JA4Generator
}

/// Generate JA4 fingerprint from ClientHello and connection info
///
/// # Arguments
/// * `generator` - JA4 generator handle
/// * `client_hello` - ClientHello packet data
/// * `is_quic` - Whether this is a QUIC connection (1) or TCP (0)
/// * `fingerprint` - Output buffer for fingerprint (will be allocated)
///
/// # Returns
/// * Result code
#[no_mangle]
pub extern "C" fn utls_ja4_generate(
    generator: *mut JA4Generator,
    client_hello: BetanetBuffer,
    is_quic: c_int,
    fingerprint: *mut BetanetBuffer,
) -> BetanetResult {
    if generator.is_null() || fingerprint.is_null() {
        return BetanetResult::InvalidArgument;
    }

    let generator_ref = unsafe { &mut *(generator as *mut betanet_utls::JA4Generator) };

    let client_hello_data = unsafe {
        if client_hello.data.is_null() {
            return BetanetResult::InvalidArgument;
        }
        client_hello.as_slice()
    };

    let protocol = if is_quic != 0 { "q" } else { "t" };

    match generator_ref.generate_fingerprint(client_hello_data, protocol) {
        Ok(fp) => {
            let fp_bytes = fp.as_bytes().to_vec();
            unsafe {
                *fingerprint = BetanetBuffer::from_vec(fp_bytes);
            }
            BetanetResult::Success
        }
        Err(_) => BetanetResult::ParseError,
    }
}

/// Free JA4 generator
///
/// # Arguments
/// * `generator` - Generator handle to free
#[no_mangle]
pub extern "C" fn utls_ja4_generator_free(generator: *mut JA4Generator) {
    if !generator.is_null() {
        unsafe {
            let _ = Box::from_raw(generator as *mut betanet_utls::JA4Generator);
        }
    }
}

/// Create uTLS ClientHello template
///
/// # Arguments
/// * `browser_type` - Browser type string (null-terminated)
///
/// # Returns
/// * Template handle on success, null on failure
#[no_mangle]
pub extern "C" fn utls_template_create(browser_type: *const c_char) -> *mut UTLSTemplate {
    if browser_type.is_null() {
        return ptr::null_mut();
    }

    let browser_str = unsafe {
        match CStr::from_ptr(browser_type).to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        }
    };

    let template = match ClientHelloTemplate::for_browser(browser_str) {
        Ok(t) => t,
        Err(_) => return ptr::null_mut(),
    };

    Box::into_raw(Box::new(template)) as *mut UTLSTemplate
}

/// Generate ClientHello from template
///
/// # Arguments
/// * `template` - Template handle
/// * `server_name` - SNI server name (null-terminated)
/// * `client_hello` - Output buffer for ClientHello (will be allocated)
///
/// # Returns
/// * Result code
#[no_mangle]
pub extern "C" fn utls_template_generate_client_hello(
    template: *mut UTLSTemplate,
    server_name: *const c_char,
    client_hello: *mut BetanetBuffer,
) -> BetanetResult {
    if template.is_null() || client_hello.is_null() {
        return BetanetResult::InvalidArgument;
    }

    let template_ref = unsafe { &mut *(template as *mut ClientHelloTemplate) };

    let server_name_str = if server_name.is_null() {
        None
    } else {
        unsafe {
            match CStr::from_ptr(server_name).to_str() {
                Ok(s) => Some(s),
                Err(_) => return BetanetResult::InvalidArgument,
            }
        }
    };

    match template_ref.generate_client_hello(server_name_str) {
        Ok(ch_data) => {
            unsafe {
                *client_hello = BetanetBuffer::from_vec(ch_data);
            }
            BetanetResult::Success
        }
        Err(_) => BetanetResult::InternalError,
    }
}

/// Free uTLS template
///
/// # Arguments
/// * `template` - Template handle to free
#[no_mangle]
pub extern "C" fn utls_template_free(template: *mut UTLSTemplate) {
    if !template.is_null() {
        unsafe {
            let _ = Box::from_raw(template as *mut ClientHelloTemplate);
        }
    }
}

/// Perform JA3/JA4 self-test
///
/// # Returns
/// * 1 if self-test passes, 0 if it fails
#[no_mangle]
pub extern "C" fn utls_self_test() -> c_int {
    match betanet_utls::run_self_test() {
        Ok(true) => 1,
        _ => 0,
    }
}
