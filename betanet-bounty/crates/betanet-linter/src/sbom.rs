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

        let spdx = json!({
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": metadata.workspace_root.file_name().unwrap_or_default(),
            "documentNamespace": format!("https://example.com/{}",
                metadata.workspace_root.file_name().unwrap_or_default()),
            "packages": metadata.packages.iter().map(|pkg| {
                json!({
                    "SPDXID": format!("SPDXRef-{}", pkg.name),
                    "name": pkg.name,
                    "version": pkg.version.to_string(),
                    "downloadLocation": "NOASSERTION",
                    "filesAnalyzed": false,
                    "copyrightText": "NOASSERTION"
                })
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
