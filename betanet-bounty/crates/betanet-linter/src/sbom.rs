//! SBOM (Software Bill of Materials) generation

use std::path::Path;
use serde_json::json;
use cargo_metadata::MetadataCommand;
use crate::Result;

/// SBOM generator
pub struct SbomGenerator;

impl SbomGenerator {
    /// Create new SBOM generator
    pub fn new() -> Self {
        Self
    }

    /// Generate SBOM
    pub async fn generate(&self, directory: &Path, format: &str) -> Result<String> {
        match format {
            "spdx" => self.generate_spdx(directory).await,
            "cyclonedx" => self.generate_cyclonedx(directory).await,
            _ => Err(crate::LinterError::Config(format!("Unsupported SBOM format: {}", format))),
        }
    }

    async fn generate_spdx(&self, directory: &Path) -> Result<String> {
        let metadata = MetadataCommand::new()
            .manifest_path(directory.join("Cargo.toml"))
            .exec()
            .map_err(|e| crate::LinterError::Parse(e.to_string()))?;

        let creation_time = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();

        let spdx = json!({
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": format!("Betanet SBOM - {}",
                metadata.workspace_root.file_name().unwrap_or_default()),
            "documentNamespace": format!("https://betanet.org/sbom/{}-{}",
                metadata.workspace_root.file_name().unwrap_or_default(),
                uuid::Uuid::new_v4()),
            "creator": "Tool: betanet-linter",
            "created": creation_time,
            "packages": metadata.packages.iter().map(|pkg| {
                json!({
                    "SPDXID": format!("SPDXRef-{}", pkg.name.replace("-", "")),
                    "name": pkg.name,
                    "version": pkg.version.to_string(),
                    "downloadLocation": pkg.repository.as_ref()
                        .map(|r| r.to_string())
                        .unwrap_or_else(|| "NOASSERTION".to_string()),
                    "filesAnalyzed": false,
                    "copyrightText": "NOASSERTION",
                    "licenseConcluded": "NOASSERTION",
                    "licenseDeclared": pkg.license.as_ref()
                        .map(|l| l.to_string())
                        .unwrap_or_else(|| "NOASSERTION".to_string()),
                    "description": pkg.description.as_ref()
                        .map(|d| d.to_string())
                        .unwrap_or_else(|| "No description".to_string()),
                    "homepage": pkg.homepage.as_ref()
                        .map(|h| h.to_string())
                        .unwrap_or_else(|| "NOASSERTION".to_string()),
                    "packageVerificationCode": {
                        "packageVerificationCodeValue": format!("{:x}",
                            pkg.name.len() as u64 + pkg.version.to_string().len() as u64)
                    }
                })
            }).collect::<Vec<_>>(),
            "relationships": metadata.packages.iter().flat_map(|pkg| {
                pkg.dependencies.iter().map(|dep| {
                    json!({
                        "spdxElementId": format!("SPDXRef-{}", pkg.name.replace("-", "")),
                        "relationshipType": "DEPENDS_ON",
                        "relatedSpdxElement": format!("SPDXRef-{}", dep.name.replace("-", ""))
                    })
                }).collect::<Vec<_>>()
            }).collect::<Vec<_>>()
        });

        Ok(serde_json::to_string_pretty(&spdx)
           .map_err(|e| crate::LinterError::Parse(e.to_string()))?)
    }

    async fn generate_cyclonedx(&self, directory: &Path) -> Result<String> {
        let metadata = MetadataCommand::new()
            .manifest_path(directory.join("Cargo.toml"))
            .exec()
            .map_err(|e| crate::LinterError::Parse(e.to_string()))?;

        let cyclonedx = json!({
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "serialNumber": format!("urn:uuid:{}", uuid::Uuid::new_v4()),
            "version": 1,
            "components": metadata.packages.iter().map(|pkg| {
                json!({
                    "type": "library",
                    "bom-ref": format!("{}@{}", pkg.name, pkg.version),
                    "name": pkg.name,
                    "version": pkg.version.to_string(),
                    "scope": "required"
                })
            }).collect::<Vec<_>>()
        });

        Ok(serde_json::to_string_pretty(&cyclonedx)
           .map_err(|e| crate::LinterError::Parse(e.to_string()))?)
    }
}

impl Default for SbomGenerator {
    fn default() -> Self {
        Self::new()
    }
}
