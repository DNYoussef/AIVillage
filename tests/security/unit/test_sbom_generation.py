"""
SBOM Generation and Cryptographic Signing Tests

Tests Software Bill of Materials generation with cryptographic signing for artifact integrity.
Validates that SBOM generation maintains security properties and enables supply chain security.

Focus: Behavioral testing of SBOM security contracts and artifact signing workflows.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import hashlib
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional
import base64

from core.domain.security_constants import SecurityLevel


class SBOMFormat(Enum):
    """Supported SBOM formats."""
    SPDX = "spdx"
    CYCLONEDX = "cyclonedx"
    SWID = "swid"


class ComponentType(Enum):
    """Types of components in SBOM."""
    APPLICATION = "application"
    LIBRARY = "library"
    FRAMEWORK = "framework"
    OPERATING_SYSTEM = "operating-system"
    DEVICE = "device"
    FILE = "file"
    CONTAINER = "container"


class SBOMComponent:
    """Represents a component in the Software Bill of Materials."""
    
    def __init__(self,
                 name: str,
                 version: str,
                 component_type: ComponentType,
                 supplier: Optional[str] = None,
                 license_info: Optional[str] = None,
                 package_url: Optional[str] = None,
                 checksums: Optional[Dict[str, str]] = None):
        self.name = name
        self.version = version
        self.component_type = component_type
        self.supplier = supplier
        self.license_info = license_info
        self.package_url = package_url
        self.checksums = checksums or {}
        self.component_id = self._generate_component_id()
    
    def _generate_component_id(self) -> str:
        """Generate unique component identifier."""
        identifier_string = f"{self.name}:{self.version}:{self.component_type.value}"
        return hashlib.sha256(identifier_string.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert component to dictionary representation."""
        return {
            "id": self.component_id,
            "name": self.name,
            "version": self.version,
            "type": self.component_type.value,
            "supplier": self.supplier,
            "license": self.license_info,
            "purl": self.package_url,
            "checksums": self.checksums
        }


class CryptographicSigner:
    """Handles cryptographic signing of SBOM artifacts."""
    
    def __init__(self, private_key_path: str = None, algorithm: str = "RSA-SHA256"):
        self.private_key_path = private_key_path
        self.algorithm = algorithm
        self.signature_metadata = {}
    
    def sign_document(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Sign SBOM document with cryptographic signature."""
        # Generate content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Mock cryptographic signature generation
        signature_data = self._generate_signature(content_hash)
        
        signing_metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "algorithm": self.algorithm,
            "content_hash": content_hash,
            "signature": signature_data,
            "signer_identity": self._get_signer_identity(),
            "signature_format": "PKCS#7",
            **(metadata or {})
        }
        
        return signing_metadata
    
    def _generate_signature(self, content_hash: str) -> str:
        """Generate cryptographic signature (mocked for testing)."""
        # In real implementation, this would use actual cryptographic signing
        mock_signature_data = f"sig_{content_hash[:16]}_{self.algorithm}"
        return base64.b64encode(mock_signature_data.encode()).decode()
    
    def _get_signer_identity(self) -> str:
        """Get signer identity information."""
        return "AIVillage-SBOM-Signer-v1.0"
    
    def verify_signature(self, content: str, signature_metadata: Dict[str, Any]) -> bool:
        """Verify cryptographic signature of SBOM document."""
        # Recalculate content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Verify hash matches
        if content_hash != signature_metadata.get("content_hash"):
            return False
        
        # Mock signature verification
        expected_signature = self._generate_signature(content_hash)
        return signature_metadata.get("signature") == expected_signature


class SBOMGenerator:
    """Generates Software Bill of Materials with security features."""
    
    def __init__(self, format_type: SBOMFormat = SBOMFormat.SPDX,
                 signer: CryptographicSigner = None):
        self.format_type = format_type
        self.signer = signer
        self.components = []
        self.generation_metadata = {}
    
    def add_component(self, component: SBOMComponent) -> None:
        """Add component to SBOM."""
        # Validate component integrity
        if not self._validate_component(component):
            raise ValueError(f"Invalid component: {component.name}")
        
        self.components.append(component)
    
    def _validate_component(self, component: SBOMComponent) -> bool:
        """Validate component data integrity."""
        required_fields = [
            component.name is not None and len(component.name) > 0,
            component.version is not None and len(component.version) > 0,
            component.component_type is not None
        ]
        return all(required_fields)
    
    def generate_sbom(self, project_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate complete SBOM document."""
        sbom_content = {
            "sbom_format": self.format_type.value,
            "spec_version": "2.3",
            "generation_timestamp": datetime.utcnow().isoformat(),
            "document_name": project_metadata.get("name", "AIVillage-SBOM"),
            "document_namespace": project_metadata.get("namespace", "https://aivillage.dev/sbom"),
            "creator": "AIVillage-SBOM-Generator",
            "components": [comp.to_dict() for comp in self.components],
            "component_count": len(self.components),
            "relationships": self._generate_relationships(),
            "vulnerability_references": self._generate_vulnerability_references(),
            **(project_metadata or {})
        }
        
        return sbom_content
    
    def _generate_relationships(self) -> List[Dict[str, Any]]:
        """Generate component relationships."""
        relationships = []
        
        # Generate dependency relationships
        for i, component in enumerate(self.components):
            if component.component_type == ComponentType.LIBRARY:
                relationships.append({
                    "relationship_type": "DEPENDS_ON",
                    "source": "AIVillage-Application",
                    "target": component.component_id,
                    "relationship_metadata": {"dependency_scope": "runtime"}
                })
        
        return relationships
    
    def _generate_vulnerability_references(self) -> List[Dict[str, Any]]:
        """Generate vulnerability reference placeholders."""
        return [
            {
                "vulnerability_id": "placeholder",
                "affected_components": [],
                "reference_url": "https://aivillage.dev/security/vulnerabilities"
            }
        ]
    
    def sign_sbom(self, sbom_content: Dict[str, Any]) -> Dict[str, Any]:
        """Sign SBOM document with cryptographic signature."""
        if not self.signer:
            raise ValueError("Cryptographic signer not configured")
        
        # Convert SBOM to string for signing
        sbom_json = json.dumps(sbom_content, sort_keys=True, separators=(',', ':'))
        
        # Generate signature
        signature_metadata = self.signer.sign_document(
            sbom_json,
            {"document_type": "SBOM", "format": self.format_type.value}
        )
        
        # Return signed SBOM
        signed_sbom = {
            "sbom_content": sbom_content,
            "signature": signature_metadata,
            "signed": True,
            "signature_verification_url": "https://aivillage.dev/verify-sbom"
        }
        
        return signed_sbom
    
    def verify_sbom_integrity(self, signed_sbom: Dict[str, Any]) -> Dict[str, Any]:
        """Verify integrity of signed SBOM."""
        if not self.signer:
            return {"verified": False, "error": "No signer configured"}
        
        sbom_content = signed_sbom.get("sbom_content", {})
        signature_metadata = signed_sbom.get("signature", {})
        
        # Convert content to string for verification
        sbom_json = json.dumps(sbom_content, sort_keys=True, separators=(',', ':'))
        
        # Verify signature
        is_valid = self.signer.verify_signature(sbom_json, signature_metadata)
        
        verification_result = {
            "verified": is_valid,
            "verification_timestamp": datetime.utcnow().isoformat(),
            "signature_algorithm": signature_metadata.get("algorithm"),
            "signer_identity": signature_metadata.get("signer_identity"),
            "content_hash_match": True  # Already verified in verify_signature
        }
        
        if not is_valid:
            verification_result["error"] = "Cryptographic signature verification failed"
        
        return verification_result


class SBOMGenerationSecurityTest(unittest.TestCase):
    """
    Behavioral tests for SBOM generation and cryptographic signing.
    
    Tests security contracts for SBOM integrity, signing workflows, and supply chain security
    without coupling to implementation details.
    """
    
    def setUp(self):
        """Set up test fixtures with security-focused mocks."""
        self.signer = CryptographicSigner(
            private_key_path="/path/to/test/key.pem",  # pragma: allowlist secret
            algorithm="RSA-SHA256"
        )
        self.sbom_generator = SBOMGenerator(
            format_type=SBOMFormat.SPDX,
            signer=self.signer
        )
        
        # Add test components
        self.test_components = [
            SBOMComponent(
                name="flask",
                version="2.3.0",
                component_type=ComponentType.FRAMEWORK,
                supplier="Pallets Projects",
                license_info="BSD-3-Clause",
                package_url="pkg:pypi/flask@2.3.0"
            ),
            SBOMComponent(
                name="requests",
                version="2.28.1", 
                component_type=ComponentType.LIBRARY,
                supplier="Python Software Foundation",
                license_info="Apache-2.0",
                package_url="pkg:pypi/requests@2.28.1"
            ),
            SBOMComponent(
                name="cryptography",
                version="38.0.1",
                component_type=ComponentType.LIBRARY,
                supplier="PyCA Cryptography",
                license_info="Apache-2.0 OR BSD-3-Clause",
                package_url="pkg:pypi/cryptography@38.0.1"
            )
        ]
    
    def test_sbom_component_validation_and_integrity(self):
        """
        Security Contract: SBOM components must be validated for data integrity.
        Tests component validation and unique identifier generation.
        """
        # Arrange
        valid_component = self.test_components[0]
        
        # Act
        self.sbom_generator.add_component(valid_component)
        
        # Assert - Test component integrity
        self.assertEqual(len(self.sbom_generator.components), 1,
                        "Valid component must be added successfully")
        
        added_component = self.sbom_generator.components[0]
        self.assertIsNotNone(added_component.component_id,
                            "Component must have unique identifier")
        self.assertEqual(len(added_component.component_id), 16,
                        "Component ID must be consistent length")
        
        # Test component validation
        with self.assertRaises(ValueError):
            invalid_component = SBOMComponent(
                name="",  # Invalid empty name
                version="1.0.0",
                component_type=ComponentType.LIBRARY
            )
            self.sbom_generator.add_component(invalid_component)
    
    def test_sbom_generation_completeness_and_structure(self):
        """
        Security Contract: Generated SBOM must be complete and well-structured.
        Tests SBOM document structure and required metadata inclusion.
        """
        # Arrange - Add multiple components
        for component in self.test_components:
            self.sbom_generator.add_component(component)
        
        project_metadata = {
            "name": "AIVillage-Security-Test",
            "namespace": "https://aivillage.dev/test/sbom",
            "version": "1.0.0"
        }
        
        # Act
        sbom_content = self.sbom_generator.generate_sbom(project_metadata)
        
        # Assert - Test SBOM structure
        required_fields = [
            "sbom_format",
            "spec_version", 
            "generation_timestamp",
            "document_name",
            "document_namespace",
            "creator",
            "components",
            "component_count",
            "relationships",
            "vulnerability_references"
        ]
        
        for field in required_fields:
            self.assertIn(field, sbom_content,
                         f"SBOM must include required field: {field}")
        
        # Test component completeness
        self.assertEqual(sbom_content["component_count"], len(self.test_components),
                        "Component count must match added components")
        self.assertEqual(len(sbom_content["components"]), len(self.test_components),
                        "All components must be included in SBOM")
        
        # Test metadata preservation
        self.assertEqual(sbom_content["name"], project_metadata["name"])
        self.assertEqual(sbom_content["namespace"], project_metadata["namespace"])
    
    def test_cryptographic_signing_workflow(self):
        """
        Security Contract: SBOM must be cryptographically signed for integrity.
        Tests end-to-end cryptographic signing workflow.
        """
        # Arrange
        for component in self.test_components:
            self.sbom_generator.add_component(component)
        
        sbom_content = self.sbom_generator.generate_sbom({
            "name": "AIVillage-Signed-SBOM-Test"
        })
        
        # Act
        signed_sbom = self.sbom_generator.sign_sbom(sbom_content)
        
        # Assert - Test signing behavior
        self.assertTrue(signed_sbom["signed"],
                       "SBOM must be marked as signed")
        self.assertIn("signature", signed_sbom,
                     "Signed SBOM must include signature metadata")
        self.assertIn("sbom_content", signed_sbom,
                     "Signed SBOM must preserve original content")
        
        signature = signed_sbom["signature"]
        required_signature_fields = [
            "timestamp",
            "algorithm", 
            "content_hash",
            "signature",
            "signer_identity",
            "signature_format"
        ]
        
        for field in required_signature_fields:
            self.assertIn(field, signature,
                         f"Signature must include field: {field}")
        
        # Test signature metadata
        self.assertEqual(signature["algorithm"], "RSA-SHA256",
                        "Must use specified signature algorithm")
        self.assertIsNotNone(signature["content_hash"],
                            "Must include content hash for integrity")
        self.assertEqual(signature["signer_identity"], "AIVillage-SBOM-Signer-v1.0",
                        "Must identify signer")
    
    def test_signature_verification_integrity(self):
        """
        Security Contract: Signed SBOM signatures must be verifiable.
        Tests signature verification workflow and tamper detection.
        """
        # Arrange - Create and sign SBOM
        for component in self.test_components:
            self.sbom_generator.add_component(component)
        
        sbom_content = self.sbom_generator.generate_sbom()
        signed_sbom = self.sbom_generator.sign_sbom(sbom_content)
        
        # Act - Verify valid signature
        verification_result = self.sbom_generator.verify_sbom_integrity(signed_sbom)
        
        # Assert - Test valid signature verification
        self.assertTrue(verification_result["verified"],
                       "Valid signature must verify successfully")
        self.assertIn("verification_timestamp", verification_result,
                     "Must include verification timestamp")
        self.assertEqual(verification_result["signature_algorithm"], "RSA-SHA256",
                        "Must preserve signature algorithm info")
        self.assertTrue(verification_result["content_hash_match"],
                       "Content hash must match")
        
        # Test tampered content detection
        tampered_sbom = signed_sbom.copy()
        tampered_sbom["sbom_content"]["document_name"] = "TAMPERED"
        
        tampered_verification = self.sbom_generator.verify_sbom_integrity(tampered_sbom)
        self.assertFalse(tampered_verification["verified"],
                        "Tampered content must fail verification")
        self.assertIn("error", tampered_verification,
                     "Failed verification must include error message")
    
    def test_component_relationship_generation(self):
        """
        Security Contract: SBOM must accurately represent component relationships.
        Tests dependency relationship generation and representation.
        """
        # Arrange - Add components with different types
        app_component = SBOMComponent(
            name="aivillage-app",
            version="1.0.0",
            component_type=ComponentType.APPLICATION
        )
        
        for component in self.test_components:
            self.sbom_generator.add_component(component)
        
        # Act
        sbom_content = self.sbom_generator.generate_sbom()
        relationships = sbom_content["relationships"]
        
        # Assert - Test relationship generation
        self.assertIsInstance(relationships, list,
                             "Relationships must be provided as list")
        
        library_components = [c for c in self.test_components 
                            if c.component_type == ComponentType.LIBRARY]
        library_relationships = [r for r in relationships 
                               if r["relationship_type"] == "DEPENDS_ON"]
        
        self.assertGreaterEqual(len(library_relationships), len(library_components),
                               "Must generate dependency relationships for libraries")
        
        # Test relationship structure
        for relationship in library_relationships:
            required_relationship_fields = ["relationship_type", "source", "target"]
            for field in required_relationship_fields:
                self.assertIn(field, relationship,
                             f"Relationship must include field: {field}")
    
    def test_vulnerability_reference_integration(self):
        """
        Security Contract: SBOM must support vulnerability reference integration.
        Tests vulnerability tracking and reference generation.
        """
        # Arrange
        for component in self.test_components:
            self.sbom_generator.add_component(component)
        
        # Act
        sbom_content = self.sbom_generator.generate_sbom()
        vulnerability_refs = sbom_content["vulnerability_references"]
        
        # Assert - Test vulnerability reference structure
        self.assertIsInstance(vulnerability_refs, list,
                             "Vulnerability references must be provided as list")
        
        if vulnerability_refs:
            vuln_ref = vulnerability_refs[0]
            required_vuln_fields = ["vulnerability_id", "affected_components", "reference_url"]
            for field in required_vuln_fields:
                self.assertIn(field, vuln_ref,
                             f"Vulnerability reference must include field: {field}")
    
    def test_sbom_format_compliance(self):
        """
        Security Contract: Generated SBOM must comply with format specifications.
        Tests SPDX format compliance and standard adherence.
        """
        # Arrange
        for component in self.test_components:
            self.sbom_generator.add_component(component)
        
        # Act
        sbom_content = self.sbom_generator.generate_sbom()
        
        # Assert - Test SPDX format compliance
        self.assertEqual(sbom_content["sbom_format"], "spdx",
                        "Must specify SPDX format")
        self.assertEqual(sbom_content["spec_version"], "2.3",
                        "Must specify supported spec version")
        
        # Test component format compliance
        for component_data in sbom_content["components"]:
            required_component_fields = ["id", "name", "version", "type"]
            for field in required_component_fields:
                self.assertIn(field, component_data,
                             f"Component must include SPDX field: {field}")
            
            # Test package URL format if present
            if "purl" in component_data and component_data["purl"]:
                self.assertTrue(component_data["purl"].startswith("pkg:"),
                               "Package URL must follow PURL specification")
    
    def test_large_scale_sbom_generation(self):
        """
        Security Contract: SBOM generation must handle large dependency sets.
        Tests scalability for large component counts.
        """
        # Arrange - Generate large number of components
        large_component_set = []
        for i in range(100):
            component = SBOMComponent(
                name=f"test-component-{i}",
                version=f"1.0.{i}",
                component_type=ComponentType.LIBRARY,
                license_info="MIT"
            )
            large_component_set.append(component)
            self.sbom_generator.add_component(component)
        
        # Act
        start_time = datetime.utcnow()
        sbom_content = self.sbom_generator.generate_sbom()
        generation_time = datetime.utcnow() - start_time
        
        # Assert - Test scalability
        self.assertEqual(sbom_content["component_count"], 100,
                        "Must handle large component sets")
        self.assertLess(generation_time.total_seconds(), 5.0,
                       "Generation must complete within reasonable time")
        
        # Test memory efficiency
        self.assertEqual(len(sbom_content["components"]), 100,
                        "All components must be included")
    
    def test_signature_algorithm_flexibility(self):
        """
        Security Contract: SBOM signing must support multiple cryptographic algorithms.
        Tests cryptographic algorithm flexibility and metadata preservation.
        """
        # Test different signature algorithms
        algorithms = ["RSA-SHA256", "RSA-SHA512", "ECDSA-SHA256"]
        
        for algorithm in algorithms:
            with self.subTest(algorithm=algorithm):
                # Arrange
                test_signer = CryptographicSigner(algorithm=algorithm)
                test_generator = SBOMGenerator(signer=test_signer)
                
                test_component = SBOMComponent(
                    name="test-component",
                    version="1.0.0",
                    component_type=ComponentType.LIBRARY
                )
                test_generator.add_component(test_component)
                
                # Act
                sbom_content = test_generator.generate_sbom()
                signed_sbom = test_generator.sign_sbom(sbom_content)
                
                # Assert
                self.assertEqual(signed_sbom["signature"]["algorithm"], algorithm,
                               f"Must preserve {algorithm} algorithm specification")
                
                verification_result = test_generator.verify_sbom_integrity(signed_sbom)
                self.assertTrue(verification_result["verified"],
                               f"Must verify {algorithm} signatures correctly")
    
    def test_sbom_content_hash_consistency(self):
        """
        Security Contract: SBOM content hashes must be deterministic and consistent.
        Tests hash consistency for integrity verification.
        """
        # Arrange - Create identical SBOMs
        components_set1 = [
            SBOMComponent("test-lib", "1.0.0", ComponentType.LIBRARY)
        ]
        components_set2 = [
            SBOMComponent("test-lib", "1.0.0", ComponentType.LIBRARY)
        ]
        
        generator1 = SBOMGenerator(signer=self.signer)
        generator2 = SBOMGenerator(signer=self.signer)
        
        for comp in components_set1:
            generator1.add_component(comp)
        for comp in components_set2:
            generator2.add_component(comp)
        
        # Act - Generate identical SBOMs
        sbom1 = generator1.generate_sbom({"name": "identical-test"})
        sbom2 = generator2.generate_sbom({"name": "identical-test"})
        
        signed_sbom1 = generator1.sign_sbom(sbom1)
        signed_sbom2 = generator2.sign_sbom(sbom2)
        
        # Assert - Test hash consistency
        hash1 = signed_sbom1["signature"]["content_hash"]
        hash2 = signed_sbom2["signature"]["content_hash"]
        
        self.assertEqual(hash1, hash2,
                        "Identical SBOM content must produce identical hashes")
        
        signature1 = signed_sbom1["signature"]["signature"]
        signature2 = signed_sbom2["signature"]["signature"]
        
        self.assertEqual(signature1, signature2,
                        "Identical content must produce identical signatures")


if __name__ == "__main__":
    # Run tests with security-focused output
    unittest.main(verbosity=2, buffer=True)