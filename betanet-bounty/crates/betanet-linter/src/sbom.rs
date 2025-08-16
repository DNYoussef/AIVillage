//! SBOM (Software Bill of Materials) generation

use cargo_metadata::MetadataCommand;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path};
use walkdir::WalkDir;

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
            _ => Err(crate::LinterError::Config(format!(
                "Unsupported SBOM format: {}",
                format
            ))),
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
            "packages": metadata
                .packages
                .iter()
                .map(|pkg| {
                    let hash = compute_package_hash(pkg)?;
                    Ok(json!({
                        "SPDXID": format!("SPDXRef-{}", pkg.name.replace('-', "")),
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
                            "packageVerificationCodeValue": hash
                        }
                    }))
                })
                .collect::<Result<Vec<_>>>()?,
            "relationships": metadata.packages.iter().flat_map(|pkg| {
                pkg.dependencies.iter().map(|dep| {
                    json!({
                        "spdxElementId": format!("SPDXRef-{}", pkg.name.replace('-', "")),
                        "relationshipType": "DEPENDS_ON",
                        "relatedSpdxElement": format!("SPDXRef-{}", dep.name.replace('-', ""))
                    })
                }).collect::<Vec<_>>()
            }).collect::<Vec<_>>()
        });

        serde_json::to_string_pretty(&spdx).map_err(|e| crate::LinterError::Parse(e.to_string()))
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

        serde_json::to_string_pretty(&cyclonedx)
            .map_err(|e| crate::LinterError::Parse(e.to_string()))
    }
}

fn compute_package_hash(pkg: &cargo_metadata::Package) -> Result<String> {
    let pkg_dir = pkg
        .manifest_path
        .parent()
        .ok_or_else(|| crate::LinterError::Sbom("Missing package directory".into()))?
        .to_path_buf();

    let mut files: Vec<_> = WalkDir::new(pkg_dir)
        .into_iter()
        .filter_entry(|e| e.file_name().to_string_lossy() != "target")
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.path().to_path_buf())
        .collect();
    files.sort();

    let mut hasher = Sha256::new();
    for path in files {
        let mut file = File::open(&path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        hasher.update(buf);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

impl Default for SbomGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[tokio::test]
    async fn package_hash_is_deterministic() -> Result<()> {
        let dir = tempdir()?;
        fs::create_dir(dir.path().join("src"))?;
        fs::write(dir.path().join("src/lib.rs"), "fn main() {}\n")?;
        fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"test_pkg\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )?;

        let generator = SbomGenerator::new();
        let sbom1 = generator.generate(dir.path(), "spdx").await?;
        let sbom2 = generator.generate(dir.path(), "spdx").await?;

        let json1: serde_json::Value =
            serde_json::from_str(&sbom1).map_err(|e| crate::LinterError::Parse(e.to_string()))?;
        let json2: serde_json::Value =
            serde_json::from_str(&sbom2).map_err(|e| crate::LinterError::Parse(e.to_string()))?;

        let hash1 = json1["packages"][0]["packageVerificationCode"]["packageVerificationCodeValue"]
            .as_str()
            .unwrap()
            .to_string();
        let hash2 = json2["packages"][0]["packageVerificationCode"]["packageVerificationCodeValue"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(hash1, hash2);

        fs::write(
            dir.path().join("src/lib.rs"),
            "fn main() { println!(\"hi\"); }\n",
        )?;
        let sbom3 = generator.generate(dir.path(), "spdx").await?;
        let json3: serde_json::Value =
            serde_json::from_str(&sbom3).map_err(|e| crate::LinterError::Parse(e.to_string()))?;
        let hash3 = json3["packages"][0]["packageVerificationCode"]["packageVerificationCodeValue"]
            .as_str()
            .unwrap()
            .to_string();
        assert_ne!(hash1, hash3);

        Ok(())
    }
}
