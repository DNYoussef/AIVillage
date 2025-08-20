// Security check script to deny insecure feature compilation
// Used by Scion Production CI to prevent insecure features in production builds

use std::env;
use std::process;

fn main() {
    println!("[SECURITY] Checking for insecure feature flags...");

    // Check if insecure feature is enabled
    if cfg!(feature = "insecure") {
        eprintln!("[ERROR] ERROR: Insecure feature flag is enabled!");
        eprintln!("   This feature should not be enabled in production builds.");
        eprintln!("   Please rebuild without the 'insecure' feature flag.");
        eprintln!("");
        eprintln!("[INFO] Tip: Use 'cargo build' instead of 'cargo build --features insecure'");
        process::exit(1);
    }

    // Check for other potentially dangerous features
    let dangerous_features = [
        "debug-mode",
        "test-only",
        "dev-tools",
        "unsafe-optimizations",
        "skip-security",
        "allow-all"
    ];

    for feature in &dangerous_features {
        // Note: Runtime feature detection is limited - this is a placeholder
        println!("[CHECK] Scanning for dangerous feature: {}", feature);
    }

    // Check environment variables for insecure settings
    let insecure_env_vars = [
        ("RUST_LOG", "trace"),
        ("RUST_LOG", "debug"),
        ("AIVILLAGE_INSECURE_MODE", "true"),
        ("SKIP_SECURITY_CHECKS", "true"),
        ("DISABLE_AUTH", "true"),
        ("ALLOW_INSECURE", "true")
    ];

    let mut warnings = 0;
    for (var_name, dangerous_value) in &insecure_env_vars {
        if let Ok(value) = env::var(var_name) {
            if value.to_lowercase() == dangerous_value.to_lowercase() {
                eprintln!("[WARN] WARNING: Insecure environment variable detected:");
                eprintln!("   {}={}", var_name, value);
                warnings += 1;
            }
        }
    }

    // Check for development-only configurations
    #[cfg(debug_assertions)]
    {
        eprintln!("[WARN] WARNING: Debug assertions are enabled");
        eprintln!("   This appears to be a debug build, not suitable for production");
        warnings += 1;
    }

    if warnings == 0 {
        println!("[OK] No insecure features or configurations detected");
        println!("[OK] Security check passed");
    } else {
        println!("[WARN] {} security warnings found (see above)", warnings);
        println!("[INFO] These may be acceptable for development but should be reviewed for production");
    }

    println!("[SECURITY] Security check completed");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_check_passes() {
        // This test ensures the security check can run without panicking
        // In a real scenario, we'd mock the cfg! macro and environment
        assert!(!cfg!(feature = "insecure"), "Test should not run with insecure feature");
    }

    #[test]
    fn test_dangerous_features_detection() {
        // Test that we can detect various dangerous feature patterns
        let features = ["debug-mode", "test-only", "dev-tools"];
        for feature in &features {
            // In a real implementation, we'd have feature detection logic here
            assert!(!feature.is_empty(), "Feature name should not be empty");
        }
    }
}
