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

    /// Generate enhanced SBOM with additional metadata
    pub async fn generate_enhanced(&self, directory: &Path, format: &str, include_dev: bool) -> Result<String> {
        match format {
            "spdx" => self.generate_enhanced_spdx(directory, include_dev).await,
            "cyclonedx" => self.generate_enhanced_cyclonedx(directory, include_dev).await,
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

    /// Generate enhanced SPDX format with security and compliance metadata
    async fn generate_enhanced_spdx(&self, directory: &Path, include_dev: bool) -> Result<String> {
        let metadata = MetadataCommand::new()
            .manifest_path(directory.join("Cargo.toml"))
            .exec()
            .map_err(|e| crate::LinterError::Parse(e.to_string()))?;

        let creation_time = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
        let document_uuid = uuid::Uuid::new_v4();

        // Filter packages based on include_dev flag
        let packages: Vec<_> = if include_dev {
            metadata.packages.iter().collect()
        } else {
            metadata.packages.iter()
                .filter(|pkg| !metadata.workspace_members.is_empty() && 
                    metadata.workspace_members.iter().any(|member| member.repr.contains(&pkg.name)))
                .collect()
        };

        let mut sbom_packages = Vec::new();
        let mut relationships = Vec::new();

        for pkg in &packages {
            let hash = compute_package_hash(pkg)?;
            let security_info = analyze_package_security(pkg).await?;
            
            sbom_packages.push(json!({
                "SPDXID": format!("SPDXRef-{}", sanitize_spdx_id(&pkg.name)),
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
                },
                "annotations": [{
                    "annotationType": "REVIEW",
                    "annotator": "Tool: betanet-linter",
                    "annotationDate": creation_time.clone(),
                    "annotationComment": format!("Security analysis: {}", 
                        serde_json::to_string(&security_info).unwrap_or_default())
                }],
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": format!("pkg:cargo/{}@{}", pkg.name, pkg.version)
                    }
                ]
            }));

            // Add dependency relationships
            for dep in &pkg.dependencies {
                relationships.push(json!({
                    "spdxElementId": format!("SPDXRef-{}", sanitize_spdx_id(&pkg.name)),
                    "relationshipType": "DEPENDS_ON",
                    "relatedSpdxElement": format!("SPDXRef-{}", sanitize_spdx_id(&dep.name))
                }));
            }
        }

        let spdx = json!({
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": format!("Betanet SBOM - {}",
                metadata.workspace_root.file_name().unwrap_or_default()),
            "documentNamespace": format!("https://betanet.org/sbom/{}-{}",
                metadata.workspace_root.file_name().unwrap_or_default(),
                document_uuid),
            "creators": [
                "Tool: betanet-linter-v1.0.0",
                format!("Organization: Betanet Project")
            ],
            "created": creation_time,
            "packages": sbom_packages,
            "relationships": relationships,
            "annotations": [{
                "annotationType": "REVIEW",
                "annotator": "Tool: betanet-linter",
                "annotationDate": creation_time.clone(),
                "annotationComment": format!("Generated SBOM for §11 compliance verification. {} packages analyzed, dev dependencies: {}",
                    packages.len(), if include_dev { "included" } else { "excluded" })
            }]
        });

        serde_json::to_string_pretty(&spdx).map_err(|e| crate::LinterError::Parse(e.to_string()))
    }

    /// Generate enhanced CycloneDX format with security and compliance metadata
    async fn generate_enhanced_cyclonedx(&self, directory: &Path, include_dev: bool) -> Result<String> {
        let metadata = MetadataCommand::new()
            .manifest_path(directory.join("Cargo.toml"))
            .exec()
            .map_err(|e| crate::LinterError::Parse(e.to_string()))?;

        let packages: Vec<_> = if include_dev {
            metadata.packages.iter().collect()
        } else {
            metadata.packages.iter()
                .filter(|pkg| !metadata.workspace_members.is_empty() && 
                    metadata.workspace_members.iter().any(|member| member.repr.contains(&pkg.name)))
                .collect()
        };

        let mut components = Vec::new();
        let mut dependencies = Vec::new();

        for pkg in &packages {
            let security_info = analyze_package_security(pkg).await?;
            
            components.push(json!({
                "type": "library",
                "bom-ref": format!("{}@{}", pkg.name, pkg.version),
                "name": pkg.name,
                "version": pkg.version.to_string(),
                "scope": if include_dev { "optional" } else { "required" },
                "description": pkg.description.as_ref()
                    .map(|d| d.to_string())
                    .unwrap_or_else(|| "No description".to_string()),
                "licenses": pkg.license.as_ref().map(|l| vec![{
                    "license": {
                        "id": l.to_string()
                    }
                }]).unwrap_or_default(),
                "externalReferences": [
                    {
                        "type": "vcs",
                        "url": pkg.repository.as_ref()
                            .map(|r| r.to_string())
                            .unwrap_or_else(|| "".to_string())
                    },
                    {
                        "type": "website",
                        "url": pkg.homepage.as_ref()
                            .map(|h| h.to_string())
                            .unwrap_or_else(|| "".to_string())
                    }
                ],
                "evidence": {
                    "identity": {
                        "field": "purl",
                        "confidence": 1.0,
                        "methods": [{
                            "technique": "instrumentation",
                            "confidence": 1.0,
                            "value": format!("pkg:cargo/{}@{}", pkg.name, pkg.version)
                        }]
                    }
                },
                "properties": [
                    {
                        "name": "betanet:security-analysis",
                        "value": serde_json::to_string(&security_info).unwrap_or_default()
                    },
                    {
                        "name": "betanet:compliance-status",
                        "value": "analyzed"
                    }
                ]
            }));

            if !pkg.dependencies.is_empty() {
                dependencies.push(json!({
                    "ref": format!("{}@{}", pkg.name, pkg.version),
                    "dependsOn": pkg.dependencies.iter().map(|dep| {
                        format!("{}@{}", dep.name, dep.version_req)
                    }).collect::<Vec<_>>()
                }));
            }
        }

        let cyclonedx = json!({
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "serialNumber": format!("urn:uuid:{}", uuid::Uuid::new_v4()),
            "version": 1,
            "metadata": {
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "tools": [{
                    "vendor": "Betanet Project",
                    "name": "betanet-linter",
                    "version": "1.0.0"
                }],
                "component": {
                    "type": "application",
                    "name": metadata.workspace_root.file_name().unwrap_or_default(),
                    "description": "Betanet protocol implementation"
                },
                "properties": [
                    {
                        "name": "betanet:sbom-type",
                        "value": if include_dev { "complete" } else { "production" }
                    },
                    {
                        "name": "betanet:compliance-target",
                        "value": "§11-specification"
                    }
                ]
            },
            "components": components,
            "dependencies": dependencies
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

/// Sanitize package name for SPDX ID format
fn sanitize_spdx_id(name: &str) -> String {
    name.replace('-', "").replace('_', "").replace('.', "")
}

/// Analyze package for security information
async fn analyze_package_security(pkg: &cargo_metadata::Package) -> Result<serde_json::Value> {
    // Basic security analysis - in practice this could integrate with vulnerability databases
    let mut security_info = serde_json::Map::new();
    
    // Check for known high-risk patterns
    let high_risk_keywords = ["unsafe", "ffi", "sys", "raw"];
    let has_high_risk = high_risk_keywords.iter().any(|keyword| {
        pkg.name.contains(keyword) || 
        pkg.description.as_ref().map_or(false, |desc| desc.to_lowercase().contains(keyword))
    });
    
    security_info.insert("risk_level".to_string(), 
        if has_high_risk { 
            serde_json::Value::String("medium".to_string()) 
        } else { 
            serde_json::Value::String("low".to_string()) 
        });
    
    security_info.insert("analysis_date".to_string(), 
        serde_json::Value::String(chrono::Utc::now().to_rfc3339()));
    
    // Check license for security implications
    if let Some(license) = &pkg.license {
        let permissive_licenses = ["MIT", "Apache-2.0", "BSD-3-Clause", "ISC"];
        let is_permissive = permissive_licenses.iter().any(|lic| license.contains(lic));
        security_info.insert("license_risk".to_string(),
            if is_permissive {
                serde_json::Value::String("low".to_string())
            } else {
                serde_json::Value::String("review_required".to_string())
            });
    }
    
    // Repository analysis
    if let Some(repo) = &pkg.repository {
        security_info.insert("has_source_repo".to_string(), serde_json::Value::Bool(true));
        security_info.insert("repo_url".to_string(), serde_json::Value::String(repo.clone()));
    } else {
        security_info.insert("has_source_repo".to_string(), serde_json::Value::Bool(false));
        security_info.insert("repo_risk".to_string(), 
            serde_json::Value::String("no_source_repo".to_string()));
    }
    
    Ok(serde_json::Value::Object(security_info))
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
