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
        } => {
            info!("Linting directory: {:?}", directory);

            let severity_level = match severity.as_str() {
                "critical" => SeverityLevel::Critical,
                "error" => SeverityLevel::Error,
                "warning" => SeverityLevel::Warning,
                _ => SeverityLevel::Info,
            };

            let config = LinterConfig {
                target_dir: directory,
                enable_all_checks: all_checks,
                generate_sbom: false,
                output_format: output_format.clone(),
                severity_level,
            };

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

        Commands::Check { rule, directory } => {
            info!("Checking rule '{}' in directory: {:?}", rule, directory);

            let config = LinterConfig {
                target_dir: directory,
                enable_all_checks: false,
                generate_sbom: false,
                output_format,
                severity_level: SeverityLevel::Info,
            };

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
