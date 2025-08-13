use clap::Parser;
use serde::Serialize;
use std::fs;
use std::path::Path;
use std::process::Command;

#[derive(Parser)]
struct Args {
    /// Path to the Betanet binary or config file
    target: String,
    /// Optional path to explicit config file
    #[arg(long)]
    config: Option<String>,
    /// Output path for JSON report
    #[arg(long)]
    report: String,
    /// Output path for CycloneDX SBOM
    #[arg(long)]
    sbom: String,
}

#[derive(Serialize)]
struct CheckResult {
    passed: bool,
    message: String,
}

#[derive(Serialize)]
struct Report {
    nonce_schedule: CheckResult,
    replay_window: CheckResult,
    per_origin_param_match: CheckResult,
    quic_tcp_fallback: CheckResult,
    rekey_thresholds: CheckResult,
    cover_traffic_budget: CheckResult,
    insecure_features_disabled: CheckResult,
    no_placeholders_release: CheckResult,
}

fn check_has(haystack: &str, needle: &str) -> CheckResult {
    let passed = haystack.contains(needle);
    let message = if passed { "present" } else { "missing" };
    CheckResult {
        passed,
        message: message.to_string(),
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let cfg_content = if let Some(cfg) = args.config.as_ref() {
        fs::read_to_string(cfg).unwrap_or_default()
    } else {
        fs::read_to_string(&args.target).unwrap_or_default()
    };

    let binary_bytes = fs::read(&args.target).unwrap_or_default();
    let placeholder_found = String::from_utf8_lossy(&binary_bytes).contains("TODO");

    let report = Report {
        nonce_schedule: check_has(&cfg_content, "nonce_schedule"),
        replay_window: check_has(&cfg_content, "replay_window"),
        per_origin_param_match: check_has(&cfg_content, "per_origin_param_match"),
        quic_tcp_fallback: check_has(&cfg_content, "quic_tcp_fallback"),
        rekey_thresholds: check_has(&cfg_content, "rekey_threshold"),
        cover_traffic_budget: check_has(&cfg_content, "cover_traffic_budget"),
        insecure_features_disabled: check_has(&cfg_content, "insecure_features=false"),
        no_placeholders_release: CheckResult {
            passed: !placeholder_found,
            message: if placeholder_found {
                "placeholder found".into()
            } else {
                "ok".into()
            },
        },
    };

    if let Some(parent) = Path::new(&args.report).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.report, serde_json::to_string_pretty(&report)?)?;

    // Generate SBOM via helper script
    Command::new("python")
        .arg("tools/sbom/generate.py")
        .arg(&args.target)
        .arg(&args.sbom)
        .status()?;

    Ok(())
}
