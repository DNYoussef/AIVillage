//! Minimal C FFI helpers for libbetanet.
//! Currently this exposes only a version query used by the C API layer.

/// Return library version for FFI consumers.
#[no_mangle]
pub extern "C" fn betanet_version() -> *const ::std::os::raw::c_char {
    const VERSION: &str = env!("CARGO_PKG_VERSION");
    VERSION.as_ptr() as *const _
}
