//! Linter FFI bindings

use crate::common::{BetanetResult, BetanetHandle, BetanetBuffer};
use std::os::raw::{c_char, c_int, c_uint};
use std::ffi::{CStr, CString};
use std::ptr;
use betanet_linter::{Linter, LinterConfig, SeverityLevel};
use crate::common::runtime;

/// Linter handle
pub type LinterHandle = BetanetHandle;

/// Linter configuration structure
#[repr(C)]
pub struct LinterConfigFFI {
    /// Target directory to lint (null-terminated string)
    pub target_dir: *const c_char,
    /// Minimum severity level (0=Info, 1=Warning, 2=Error, 3=Critical)
    pub min_severity: c_uint,
    /// Enable all checks
    pub all_checks: c_int,
}

/// Linter results structure
#[repr(C)]
pub struct LinterResultsFFI {
    /// Number of files checked
    pub files_checked: c_uint,
    /// Number of rules executed
    pub rules_executed: c_uint,
    /// Number of critical issues
    pub critical_issues: c_uint,
    /// Number of error issues
    pub error_issues: c_uint,
    /// Number of warning issues
    pub warning_issues: c_uint,
    /// Number of info issues
    pub info_issues: c_uint,
}

/// Create linter with configuration
///
/// # Arguments
/// * `config` - Linter configuration
///
/// # Returns
/// * Linter handle on success, null on failure
#[no_mangle]
pub extern "C" fn linter_create(config: *const LinterConfigFFI) -> *mut LinterHandle {
    let config_data = if config.is_null() {
        LinterConfig::default()
    } else {
        let config_ref = unsafe { &*config };

        let target_dir = if config_ref.target_dir.is_null() {
            ".".to_string()
        } else {
            unsafe {
                match CStr::from_ptr(config_ref.target_dir).to_str() {
                    Ok(s) => s.to_string(),
                    Err(_) => return ptr::null_mut(),
                }
            }
        };

        let min_severity = match config_ref.min_severity {
            0 => SeverityLevel::Info,
            1 => SeverityLevel::Warning,
            2 => SeverityLevel::Error,
            3 => SeverityLevel::Critical,
            _ => SeverityLevel::Warning,
        };

        LinterConfig {
            target_dir: target_dir.into(),
            min_severity,
            all_checks: config_ref.all_checks != 0,
        }
    };

    let linter = Linter::new(config_data);
    Box::into_raw(Box::new(linter)) as *mut LinterHandle
}

/// Run linter checks
///
/// # Arguments
/// * `linter` - Linter handle
/// * `results` - Output results structure owned by the caller
///   which will be populated on success
///
/// # Returns
/// * Result code
#[no_mangle]
pub extern "C" fn linter_run(
    linter: *mut LinterHandle,
    results: *mut LinterResultsFFI,
) -> BetanetResult {
    if linter.is_null() || results.is_null() {
        return BetanetResult::InvalidArgument;
    }

    let linter_ref = unsafe { &mut *(linter as *mut Linter) };

    // Execute using shared async runtime
    let lint_results = match runtime().block_on(linter_ref.run()) {
        Ok(r) => r,
        Err(_) => return BetanetResult::InternalError,
    };

    let mut critical = 0;
    let mut error = 0;
    let mut warning = 0;
    let mut info = 0;

    for issue in &lint_results.issues {
        match issue.severity {
            SeverityLevel::Critical => critical += 1,
            SeverityLevel::Error => error += 1,
            SeverityLevel::Warning => warning += 1,
            SeverityLevel::Info => info += 1,
        }
    }

    unsafe {
        *results = LinterResultsFFI {
            files_checked: lint_results.files_checked as c_uint,
            rules_executed: lint_results.rules_executed as c_uint,
            critical_issues: critical,
            error_issues: error,
            warning_issues: warning,
            info_issues: info,
        };
    }

    BetanetResult::Success
}

/// Check specific linting rule
///
/// # Arguments
/// * `linter` - Linter handle
/// * `rule_name` - Rule name to check (null-terminated string)
/// * `results` - Output results structure owned by the caller and
///   populated on success
///
/// # Returns
/// * Result code
#[no_mangle]
pub extern "C" fn linter_check_rule(
    linter: *mut LinterHandle,
    rule_name: *const c_char,
    results: *mut LinterResultsFFI,
) -> BetanetResult {
    if linter.is_null() || rule_name.is_null() || results.is_null() {
        return BetanetResult::InvalidArgument;
    }

    let linter_ref = unsafe { &mut *(linter as *mut Linter) };

    let rule_str = unsafe {
        match CStr::from_ptr(rule_name).to_str() {
            Ok(s) => s,
            Err(_) => return BetanetResult::InvalidArgument,
        }
    };

    // Execute using shared async runtime
    let lint_results = match runtime().block_on(linter_ref.check_rule(rule_str)) {
        Ok(r) => r,
        Err(_) => return BetanetResult::InternalError,
    };

    let mut critical = 0;
    let mut error = 0;
    let mut warning = 0;
    let mut info = 0;

    for issue in &lint_results.issues {
        match issue.severity {
            SeverityLevel::Critical => critical += 1,
            SeverityLevel::Error => error += 1,
            SeverityLevel::Warning => warning += 1,
            SeverityLevel::Info => info += 1,
        }
    }

    unsafe {
        *results = LinterResultsFFI {
            files_checked: lint_results.files_checked as c_uint,
            rules_executed: lint_results.rules_executed as c_uint,
            critical_issues: critical,
            error_issues: error,
            warning_issues: warning,
            info_issues: info,
        };
    }

    BetanetResult::Success
}

/// Generate SBOM
///
/// # Arguments
/// * `directory` - Directory to analyze (null-terminated string)
/// * `format` - SBOM format ("spdx" or "cyclonedx", null-terminated string)
/// * `output` - Output buffer for SBOM JSON. The buffer is allocated by the
///   library and the caller must release it with `betanet_buffer_free`.
///
/// # Returns
/// * Result code
#[no_mangle]
pub extern "C" fn linter_generate_sbom(
    directory: *const c_char,
    format: *const c_char,
    output: *mut BetanetBuffer,
) -> BetanetResult {
    if output.is_null() {
        return BetanetResult::InvalidArgument;
    }

    let dir_str = if directory.is_null() {
        "."
    } else {
        unsafe {
            match CStr::from_ptr(directory).to_str() {
                Ok(s) => s,
                Err(_) => return BetanetResult::InvalidArgument,
            }
        }
    };

    let format_str = if format.is_null() {
        "spdx"
    } else {
        unsafe {
            match CStr::from_ptr(format).to_str() {
                Ok(s) => s,
                Err(_) => return BetanetResult::InvalidArgument,
            }
        }
    };

    let generator = betanet_linter::SbomGenerator::new();
    let dir_path = std::path::Path::new(dir_str);

    // Execute using shared async runtime
    let sbom_json = match runtime().block_on(generator.generate(dir_path, format_str)) {
        Ok(json) => json,
        Err(_) => return BetanetResult::InternalError,
    };

    unsafe {
        *output = BetanetBuffer::from_vec(sbom_json.into_bytes());
    }

    BetanetResult::Success
}

/// Free linter
///
/// # Arguments
/// * `linter` - Linter handle to free
#[no_mangle]
pub extern "C" fn linter_free(linter: *mut LinterHandle) {
    if !linter.is_null() {
        unsafe {
            let _ = Box::from_raw(linter as *mut Linter);
        }
    }
}
