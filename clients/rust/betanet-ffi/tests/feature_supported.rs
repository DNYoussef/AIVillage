use betanet_ffi::{betanet_feature_supported, BetanetResult};
use std::ffi::CString;

#[test]
fn feature_supported_known_feature() {
    let feature = CString::new("ffi_demo").unwrap();
    let result = betanet_feature_supported(feature.as_ptr());
    assert_eq!(result, BetanetResult::Success);
}

#[test]
fn feature_supported_unknown_feature() {
    let feature = CString::new("unknown").unwrap();
    let result = betanet_feature_supported(feature.as_ptr());
    assert_eq!(result, BetanetResult::NotSupported);
}

#[test]
fn feature_supported_null_pointer() {
    let result = betanet_feature_supported(std::ptr::null());
    assert_eq!(result, BetanetResult::InvalidArgument);
}

#[test]
fn result_code_mapping() {
    assert_eq!(BetanetResult::Success as i32, 0);
    assert_eq!(BetanetResult::InvalidArgument as i32, -1);
    assert_eq!(BetanetResult::NotSupported as i32, -7);
}
