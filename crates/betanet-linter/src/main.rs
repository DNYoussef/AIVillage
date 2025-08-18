//! Betanet Linter CLI

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use betanet_linter::{
    lint::Linter, report::ReportGenerator, LinterConfig, OutputFormat, Result, SeverityLevel,
};

/// Betanet Linter CLI
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Subcommand to run
    #[command(subcommand)]
    command: Commands,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Output format
    #[arg(short, long, default_value = "text")]
    format: String,
}

/// Available commands
#[derive(Subcommand)]
enum Commands {
    /// Lint Betanet components
    Lint {
        /// Directory to lint
        #[arg(short, long, default_value = ".")]
        directory: PathBuf,

        /// Minimum severity level to report
        #[arg(short, long, default_value = "info")]
        severity: String,

        /// Enable all checks
        #[arg(long)]
        all_checks: bool,

        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Additional paths to ignore
        #[arg(long, value_name = "PATH")]
        ignore: Vec<PathBuf>,
    },

    /// Generate SBOM
    Sbom {
        /// Directory to analyze
        #[arg(short, long, default_value = ".")]
        directory: PathBuf,

        /// Output file
        #[arg(short, long, default_value = "sbom.json")]
        output: PathBuf,

        /// SBOM format
        #[arg(short, long, default_value = "spdx")]
        format: String,
    },

    /// Check specific rule
    Check {
        /// Rule to check
        #[arg(short, long)]
        rule: String,

        /// Directory to check
        #[arg(short, long, default_value = ".")]
        directory: PathBuf,

        /// Additional paths to ignore
        #[arg(long, value_name = "PATH")]
        ignore: Vec<PathBuf>,
    },

    /// Security scan for vulnerable binaries
    SecurityScan {
        /// Directory or binary to scan
        #[arg(short, long, default_value = ".")]
        target: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Fail on any security issues
        #[arg(long)]
        fail_on_issues: bool,

        /// Additional paths to ignore
        #[arg(long, value_name = "PATH")]
        ignore: Vec<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing
    let filter = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| filter.into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let output_format = match cli.format.as_str() {
        "json" => OutputFormat::Json,
        "sarif" => OutputFormat::Sarif,
        _ => OutputFormat::Text,
    };

    match cli.command {
        Commands::Lint {
            directory,
            severity,
            all_checks,
            output,
            ignore,
        } => {
            info!("Linting directory: {:?}", directory);

            let severity_level = match severity.as_str() {
                "critical" => SeverityLevel::Critical,
                "error" => SeverityLevel::Error,
                "warning" => SeverityLevel::Warning,
                _ => SeverityLevel::Info,
            };

            let mut config = LinterConfig::default();
            config.target_dir = directory;
            config.enable_all_checks = all_checks;
            config.output_format = output_format.clone();
            config.severity_level = severity_level;
            config.ignored_paths.extend(ignore);

            let linter = Linter::new(config);
            let results = linter.run().await?;

            let generator = ReportGenerator::new(output_format);
            let report = generator.generate(&results)?;

            if let Some(output_file) = output {
                std::fs::write(output_file, report)?;
                info!("Report written to file");
            } else {
                println!("{}", report);
            }

            // Exit with error code if issues found
            if results.has_issues_above(&SeverityLevel::Error) {
                std::process::exit(1);
            }
        }

        Commands::Sbom {
            directory,
            output,
            format,
        } => {
            info!("Generating SBOM for directory: {:?}", directory);

            #[cfg(feature = "sbom")]
            {
                let sbom_generator = betanet_linter::sbom::SbomGenerator::new();
                let sbom = sbom_generator.generate(&directory, &format).await?;

                std::fs::write(output, sbom)?;
                info!("SBOM generated successfully");
            }

            #[cfg(not(feature = "sbom"))]
            {
                error!("SBOM feature not enabled");
                std::process::exit(1);
            }
        }

        Commands::Check { rule, directory, ignore } => {
            info!("Checking rule '{}' in directory: {:?}", rule, directory);

            let mut config = LinterConfig::default();
            config.target_dir = directory;
            config.enable_all_checks = false;
            config.output_format = output_format;
            config.severity_level = SeverityLevel::Info;
            config.ignored_paths.extend(ignore);

            let linter = Linter::new(config);
            let results = linter.check_rule(&rule).await?;

            if results.issues.is_empty() {
                println!("✓ Rule '{}' passed", rule);
            } else {
                println!(
                    "✗ Rule '{}' failed with {} issues",
                    rule,
                    results.issues.len()
                );
                for issue in &results.issues {
                    println!("  {}: {}", issue.severity_str(), issue.message);
                }
            }
        }

        Commands::SecurityScan { target, output, fail_on_issues, ignore } => {
            info!("Running security scan on: {:?}", target);

            let mut config = LinterConfig::default();
            config.target_dir = target;
            config.enable_all_checks = false;
            config.output_format = output_format.clone();
            config.severity_level = SeverityLevel::Info;
            config.ignored_paths.extend(ignore);

            let linter = Linter::new(config);
            let results = linter.security_scan().await?;

            let generator = ReportGenerator::new(output_format);
            let report = generator.generate(&results)?;

            if let Some(output_file) = output {
                std::fs::write(output_file, &report)?;
                info!("Security scan report written to file");
            } else {
                println!("{}", report);
            }

            // Print summary
            let (critical, errors, warnings, info) = results.summary();
            println!("\n🔍 Security Scan Summary:");
            println!("  Critical: {}", critical);
            println!("  Errors: {}", errors);
            println!("  Warnings: {}", warnings);
            println!("  Info: {}", info);
            println!("  Files scanned: {}", results.files_checked);

            if critical > 0 {
                println!("\n❌ CRITICAL SECURITY VULNERABILITIES FOUND!");
                println!("   Immediate action required - do not use these binaries in production");
                if fail_on_issues {
                    std::process::exit(2);
                }
            } else if errors > 0 {
                println!("\n⚠️  Security issues found - review and update recommended");
                if fail_on_issues {
                    std::process::exit(1);
                }
            } else {
                println!("\n✅ No critical security vulnerabilities detected");
            }
        }
    }

    Ok(())
}

// Helper trait for displaying severity
trait SeverityStr {
    fn severity_str(&self) -> &'static str;
}

impl SeverityStr for betanet_linter::LintIssue {
    fn severity_str(&self) -> &'static str {
        match self.severity {
            SeverityLevel::Critical => "CRITICAL",
            SeverityLevel::Error => "ERROR",
            SeverityLevel::Warning => "WARNING",
            SeverityLevel::Info => "INFO",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parsing() {
        let cli = Cli::try_parse_from(["betanet-linter", "lint", "--directory", "."]);
        assert!(cli.is_ok());
    }
}
