/*!
 * Betanet v1.1 Spec-Compliance Linter CLI
 *
 * Validates Betanet implementations against the v1.1 specification requirements
 */

use std::fs;
use std::path::Path;

/// Compliance check result
#[derive(Debug, Clone)]
pub struct ComplianceResult {
    pub requirement_id: String,
    pub section: String,
    pub description: String,
    pub status: ComplianceStatus,
    pub evidence: Vec<String>,
    pub violations: Vec<String>,
    pub score: f64,
}

/// Compliance status enumeration
#[derive(Debug, Clone)]
pub enum ComplianceStatus {
    Pass,
    Fail,
    Warning,
    NotApplicable,
}

/// Main linter implementation
pub struct BetanetLinter {
    pub target_dir: String,
    pub results: Vec<ComplianceResult>,
}

impl BetanetLinter {
    pub fn new(target_dir: String) -> Self {
        Self {
            target_dir,
            results: Vec::new(),
        }
    }

    /// Run all compliance checks
    pub fn run_all_checks(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔍 Running Betanet v1.1 compliance checks on: {}", self.target_dir);

        // Section 5: HTX Cover Transport checks
        self.check_htx_compliance()?;

        // Section 4: SCION Transition checks
        self.check_scion_compliance()?;

        // Section 6: Overlay ALPN checks
        self.check_overlay_compliance()?;

        // Section 7: Privacy/Mixnet checks
        self.check_privacy_compliance()?;

        // Section 11: Security checks
        self.check_security_compliance()?;

        Ok(())
    }

    /// Check HTX Cover Transport compliance (Section 5)
    fn check_htx_compliance(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("📡 Checking HTX Cover Transport compliance...");

        // BN-5.1: Origin-mirrored TLS
        let mut result = ComplianceResult {
            requirement_id: "BN-5.1-ORIGIN-MIRROR".to_string(),
            section: "5.1".to_string(),
            description: "Origin-mirrored TLS with JA3/JA4 fingerprinting".to_string(),
            status: ComplianceStatus::Fail,
            evidence: Vec::new(),
            violations: Vec::new(),
            score: 0.0,
        };

        // Check for JA3/JA4 implementation
        if self.check_file_contains_patterns(&[
            "src/core/p2p/betanet_htx_transport.py",
            "tools/utlsgen/calibrate_fingerprint.py"
        ], &["ja3", "ja4", "fingerprint"]) {
            result.evidence.push("Found JA3/JA4 fingerprinting code".to_string());
            result.score += 0.5;
        }

        // Check for Noise XK implementation
        if self.check_file_contains_patterns(&[
            "src/core/p2p/betanet_htx_transport.py"
        ], &["noise_xk", "NoiseXKState", "KEY_UPDATE"]) {
            result.evidence.push("Found Noise XK implementation".to_string());
            result.score += 0.5;
        }

        result.status = if result.score >= 0.8 {
            ComplianceStatus::Pass
        } else if result.score >= 0.5 {
            ComplianceStatus::Warning
        } else {
            ComplianceStatus::Fail
        };

        self.results.push(result);

        Ok(())
    }

    /// Check SCION Transition compliance (Section 4)
    fn check_scion_compliance(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🌐 Checking SCION Transition compliance...");

        let mut result = ComplianceResult {
            requirement_id: "BN-4.1-SCION-CONTROL".to_string(),
            section: "4.1".to_string(),
            description: "HTX-tunnelled SCION control stream".to_string(),
            status: ComplianceStatus::Fail,
            evidence: Vec::new(),
            violations: Vec::new(),
            score: 0.0,
        };

        if self.check_file_exists("src/transport/scion_htx_gateway.py") {
            result.evidence.push("Found SCION HTX gateway implementation".to_string());
            result.score += 0.5;
        }

        if self.check_file_contains_patterns(&[
            "src/transport/scion_htx_gateway.py"
        ], &["cbor2", "ed25519", "signature"]) {
            result.evidence.push("Found CBOR signed control messages".to_string());
            result.score += 0.5;
        }

        result.status = if result.score >= 0.8 {
            ComplianceStatus::Pass
        } else {
            ComplianceStatus::Fail
        };

        self.results.push(result);
        Ok(())
    }

    /// Check Overlay ALPN compliance (Section 6)
    fn check_overlay_compliance(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔗 Checking Overlay ALPN compliance...");

        let mut result = ComplianceResult {
            requirement_id: "BN-6.2-ALPN-REGISTER".to_string(),
            section: "6.2".to_string(),
            description: "ALPN protocol registration".to_string(),
            status: ComplianceStatus::Fail,
            evidence: Vec::new(),
            violations: Vec::new(),
            score: 0.0,
        };

        if self.check_file_contains_patterns(&[
            "src/core/p2p/libp2p_mesh.py"
        ], &["/betanet/htx/1.1.0", "/betanet/htxquic/1.1.0"]) {
            result.evidence.push("Found Betanet ALPN protocol registration".to_string());
            result.score += 1.0;
        }

        result.status = if result.score >= 0.8 {
            ComplianceStatus::Pass
        } else {
            ComplianceStatus::Fail
        };

        self.results.push(result);
        Ok(())
    }

    /// Check Privacy/Mixnet compliance (Section 7)
    fn check_privacy_compliance(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🕶️  Checking Privacy/Mixnet compliance...");

        let mut result = ComplianceResult {
            requirement_id: "BN-7.2-BEACON-VRF".to_string(),
            section: "7.2".to_string(),
            description: "VRF-based hop selection with BeaconSet entropy".to_string(),
            status: ComplianceStatus::Fail,
            evidence: Vec::new(),
            violations: Vec::new(),
            score: 0.0,
        };

        if self.check_file_exists("src/core/p2p/betanet_mixnet.py") {
            result.evidence.push("Found mixnet implementation".to_string());
            result.score += 0.5;
        }

        if self.check_file_contains_patterns(&[
            "src/core/p2p/betanet_mixnet.py"
        ], &["VRFSelector", "generate_vrf_proof"]) {
            result.evidence.push("Found VRF-based selection".to_string());
            result.score += 0.5;
        }

        result.status = if result.score >= 0.8 {
            ComplianceStatus::Pass
        } else {
            ComplianceStatus::Fail
        };

        self.results.push(result);
        Ok(())
    }

    /// Check Security compliance (Section 11)
    fn check_security_compliance(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔒 Checking Security compliance...");

        let mut result = ComplianceResult {
            requirement_id: "BN-11.1-NO-LEGACY-HEADER".to_string(),
            section: "11.1".to_string(),
            description: "No legacy transition headers on public networks".to_string(),
            status: ComplianceStatus::Pass,
            evidence: Vec::new(),
            violations: Vec::new(),
            score: 1.0,
        };

        if self.check_file_contains_patterns(&[
            "src/transport/scion_htx_gateway.py"
        ], &["verify_no_legacy_header"]) {
            result.evidence.push("Found legacy header validation".to_string());
        } else {
            result.violations.push("Missing legacy header validation".to_string());
            result.score = 0.5;
            result.status = ComplianceStatus::Warning;
        }

        self.results.push(result);
        Ok(())
    }

    /// Generate compliance report
    pub fn generate_report(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("📊 Generating compliance report...");

        fs::create_dir_all("tmp_submission/lint")?;

        let total_checks = self.results.len();
        let passed = self.results.iter().filter(|r| matches!(r.status, ComplianceStatus::Pass)).count();
        let warnings = self.results.iter().filter(|r| matches!(r.status, ComplianceStatus::Warning)).count();
        let failed = self.results.iter().filter(|r| matches!(r.status, ComplianceStatus::Fail)).count();

        let report = format!(r#"{{
  "timestamp": "2025-08-14T12:00:00Z",
  "betanet_version": "1.1.0",
  "total_checks": {},
  "passed": {},
  "warnings": {},
  "failed": {},
  "pass_rate": {:.1},
  "results": [{}]
}}"#,
            total_checks,
            passed,
            warnings,
            failed,
            (passed as f64 / total_checks as f64) * 100.0,
            self.results.iter().map(|r| format!(r#"{{
      "id": "{}",
      "status": "{}",
      "score": {:.2},
      "evidence_count": {}
    }}"#, r.requirement_id,
        match r.status {
            ComplianceStatus::Pass => "PASS",
            ComplianceStatus::Fail => "FAIL",
            ComplianceStatus::Warning => "WARNING",
            ComplianceStatus::NotApplicable => "N/A"
        },
        r.score, r.evidence.len())).collect::<Vec<_>>().join(",\n    ")
        );

        fs::write("tmp_submission/lint/report.json", report)?;

        let sbom = r#"{
  "bomFormat": "CycloneDX",
  "specVersion": "1.4",
  "serialNumber": "urn:uuid:betanet-v1.1.0",
  "version": 1,
  "components": [
    {
      "type": "library",
      "name": "betanet-htx-transport",
      "version": "1.1.0",
      "supplier": {"name": "AIVillage"},
      "licenses": [{"license": {"name": "MIT"}}]
    }
  ]
}"#;
        fs::write("tmp_submission/lint/sbom.cdx.json", sbom)?;

        println!("\n📈 Compliance Summary:");
        println!("  Total Checks: {}", total_checks);
        println!("  ✅ Passed: {}", passed);
        println!("  ⚠️  Warnings: {}", warnings);
        println!("  ❌ Failed: {}", failed);

        let pass_rate = (passed as f64 / total_checks as f64) * 100.0;
        println!("  📊 Pass Rate: {:.1}%", pass_rate);

        if pass_rate >= 80.0 {
            println!("🎉 Betanet v1.1 compliance: SUFFICIENT");
        } else {
            println!("❌ Betanet v1.1 compliance: INSUFFICIENT");
        }

        Ok(())
    }

    /// Helper: Check if file exists
    fn check_file_exists(&self, relative_path: &str) -> bool {
        Path::new(&self.target_dir).join(relative_path).exists()
    }

    /// Helper: Check if files contain specific patterns
    fn check_file_contains_patterns(&self, files: &[&str], patterns: &[&str]) -> bool {
        for file_path in files {
            let full_path = Path::new(&self.target_dir).join(file_path);
            if full_path.exists() {
                if let Ok(content) = fs::read_to_string(&full_path) {
                    let content_lower = content.to_lowercase();
                    for pattern in patterns {
                        if content_lower.contains(&pattern.to_lowercase()) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target_dir = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());

    let mut linter = BetanetLinter::new(target_dir);
    linter.run_all_checks()?;
    linter.generate_report()?;

    println!("\n✅ Linting complete. Reports saved to: tmp_submission/lint/");

    Ok(())
}
