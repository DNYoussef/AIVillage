#!/usr/bin/env python3
"""
Enhanced SBOM Generation and Artifact Signing Architecture
Comprehensive Software Bill of Materials generation with vulnerability scanning and artifact integrity.
"""

import asyncio
import hashlib
import json
import subprocess
import tempfile
import zipfile
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
import requests
import semver
import toml
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class ComponentType(Enum):
    """Types of software components"""
    LIBRARY = "library"
    FRAMEWORK = "framework"
    APPLICATION = "application"
    CONTAINER = "container"
    OPERATING_SYSTEM = "operating-system"
    DEVICE = "device"

class VulnerabilitySeverity(Enum):
    """Vulnerability severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class LicenseRisk(Enum):
    """License compliance risk levels"""
    HIGH = "high"        # GPL, AGPL
    MEDIUM = "medium"    # LGPL, MPL
    LOW = "low"         # Apache, MIT, BSD
    UNKNOWN = "unknown"  # Unrecognized license

@dataclass
class Vulnerability:
    """Vulnerability information"""
    cve_id: str
    severity: VulnerabilitySeverity
    cvss_score: float
    description: str
    affected_versions: List[str]
    fixed_version: Optional[str]
    published_date: datetime
    references: List[str]
    
class Component:
    """Software component with security metadata"""
    
    def __init__(self, 
                 name: str,
                 version: str,
                 component_type: ComponentType,
                 supplier: str = None,
                 licenses: List[str] = None,
                 cpe: str = None,
                 purl: str = None,
                 description: str = None):
        self.bom_ref = self._generate_bom_ref(name, version)
        self.name = name
        self.version = version
        self.component_type = component_type
        self.supplier = supplier
        self.licenses = licenses or []
        self.cpe = cpe
        self.purl = purl
        self.description = description
        self.hashes = {}
        self.vulnerabilities: List[Vulnerability] = []
        self.license_risk = LicenseRisk.UNKNOWN
        self.supply_chain_risk = 0.0  # 0-10 scale
        self.last_updated = None
        self.external_references = []
        
    def _generate_bom_ref(self, name: str, version: str) -> str:
        """Generate unique BOM reference using SHA256 for security"""
        ref = f"{name}-{version}".replace('/', '-').replace('@', '-')
        return hashlib.sha256(ref.encode()).hexdigest()[:16]  # Use SHA256 truncated for uniqueness
    
    def add_hash(self, algorithm: str, value: str):
        """Add hash for integrity verification"""
        self.hashes[algorithm] = value
    
    def add_vulnerability(self, vulnerability: Vulnerability):
        """Add vulnerability to component"""
        self.vulnerabilities.append(vulnerability)
    
    def assess_license_risk(self):
        """Assess license compliance risk"""
        high_risk_licenses = {'GPL', 'AGPL', 'GPL-2.0', 'GPL-3.0', 'AGPL-3.0'}
        medium_risk_licenses = {'LGPL', 'MPL', 'LGPL-2.1', 'LGPL-3.0', 'MPL-2.0'}
        low_risk_licenses = {'MIT', 'Apache', 'BSD', 'ISC', 'Apache-2.0', 'BSD-3-Clause', 'BSD-2-Clause'}
        
        for license_name in self.licenses:
            license_upper = license_name.upper()
            if any(hr in license_upper for hr in high_risk_licenses):
                self.license_risk = LicenseRisk.HIGH
                return
            elif any(mr in license_upper for mr in medium_risk_licenses):
                self.license_risk = LicenseRisk.MEDIUM
        
        if any(any(lr in license_name.upper() for lr in low_risk_licenses) for license_name in self.licenses):
            self.license_risk = LicenseRisk.LOW
        else:
            self.license_risk = LicenseRisk.UNKNOWN
    
    def calculate_supply_chain_risk(self):
        """Calculate supply chain risk score"""
        risk_factors = []
        
        # Vulnerability score
        if self.vulnerabilities:
            vuln_scores = [v.cvss_score for v in self.vulnerabilities]
            max_cvss = max(vuln_scores)
            risk_factors.append(max_cvss)
        
        # Age factor (older versions are riskier)
        if self.last_updated:
            days_old = (datetime.utcnow() - self.last_updated).days
            age_risk = min(10.0, days_old / 365 * 5)  # Max 5 points for age
            risk_factors.append(age_risk)
        
        # License risk
        license_risk_scores = {
            LicenseRisk.HIGH: 3.0,
            LicenseRisk.MEDIUM: 2.0,
            LicenseRisk.LOW: 0.5,
            LicenseRisk.UNKNOWN: 2.5
        }
        risk_factors.append(license_risk_scores[self.license_risk])
        
        # Supplier trust (simplified)
        trusted_suppliers = {'microsoft', 'google', 'apache', 'mozilla', 'python', 'nodejs'}
        if self.supplier and any(ts in self.supplier.lower() for ts in trusted_suppliers):
            risk_factors.append(0.5)  # Lower risk for trusted suppliers
        else:
            risk_factors.append(2.0)  # Higher risk for unknown suppliers
        
        self.supply_chain_risk = min(10.0, sum(risk_factors) / len(risk_factors) if risk_factors else 5.0)

class VulnerabilityScanner:
    """Scanner for known vulnerabilities"""
    
    def __init__(self):
        self.osv_api_base = "https://api.osv.dev/v1"
        self.nvd_api_base = "https://services.nvd.nist.gov/rest/json"
        
    async def scan_component(self, component: Component) -> List[Vulnerability]:
        """Scan component for vulnerabilities"""
        vulnerabilities = []
        
        # Query OSV database
        osv_vulns = await self._query_osv(component)
        vulnerabilities.extend(osv_vulns)
        
        # Query NVD database
        nvd_vulns = await self._query_nvd(component)
        vulnerabilities.extend(nvd_vulns)
        
        # Deduplicate by CVE ID
        seen_cves = set()
        unique_vulns = []
        for vuln in vulnerabilities:
            if vuln.cve_id not in seen_cves:
                unique_vulns.append(vuln)
                seen_cves.add(vuln.cve_id)
        
        return unique_vulns
    
    async def _query_osv(self, component: Component) -> List[Vulnerability]:
        """Query OSV database for vulnerabilities"""
        try:
            payload = {
                "package": {
                    "name": component.name,
                    "ecosystem": self._get_ecosystem(component)
                },
                "version": component.version
            }
            
            # In production, use aiohttp for async requests
            response = requests.post(f"{self.osv_api_base}/query", json=payload, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"OSV query failed for {component.name}: {response.status_code}")
                return []
            
            data = response.json()
            vulnerabilities = []
            
            for vuln_data in data.get("vulns", []):
                vuln = self._parse_osv_vulnerability(vuln_data)
                if vuln:
                    vulnerabilities.append(vuln)
            
            return vulnerabilities
            
        except Exception as e:
            logger.error(f"Error querying OSV for {component.name}: {e}")
            return []
    
    async def _query_nvd(self, component: Component) -> List[Vulnerability]:
        """Query NVD database for vulnerabilities"""
        try:
            # Query NVD by CPE if available
            if not component.cpe:
                return []
            
            params = {
                "cpeName": component.cpe,
                "resultsPerPage": 100
            }
            
            response = requests.get(f"{self.nvd_api_base}/cves/2.0", params=params, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"NVD query failed for {component.name}: {response.status_code}")
                return []
            
            data = response.json()
            vulnerabilities = []
            
            for vuln_data in data.get("vulnerabilities", []):
                vuln = self._parse_nvd_vulnerability(vuln_data)
                if vuln:
                    vulnerabilities.append(vuln)
            
            return vulnerabilities
            
        except Exception as e:
            logger.error(f"Error querying NVD for {component.name}: {e}")
            return []
    
    def _get_ecosystem(self, component: Component) -> str:
        """Determine ecosystem from component"""
        # This would be more sophisticated in production
        if component.purl:
            if "pypi" in component.purl:
                return "PyPI"
            elif "npm" in component.purl:
                return "npm"
            elif "cargo" in component.purl:
                return "crates.io"
        
        return "Unknown"
    
    def _parse_osv_vulnerability(self, vuln_data: Dict) -> Optional[Vulnerability]:
        """Parse OSV vulnerability data"""
        try:
            vuln_id = vuln_data.get("id", "")
            summary = vuln_data.get("summary", "")
            
            # Extract CVE if present
            cve_id = vuln_id if vuln_id.startswith("CVE-") else None
            for alias in vuln_data.get("aliases", []):
                if alias.startswith("CVE-"):
                    cve_id = alias
                    break
            
            if not cve_id:
                cve_id = vuln_id  # Use OSV ID as fallback
            
            # Parse severity
            severity = VulnerabilitySeverity.MEDIUM  # Default
            cvss_score = 5.0  # Default
            
            for severity_data in vuln_data.get("severity", []):
                if severity_data.get("type") == "CVSS_V3":
                    cvss_score = float(severity_data.get("score", 5.0))
                    if cvss_score >= 9.0:
                        severity = VulnerabilitySeverity.CRITICAL
                    elif cvss_score >= 7.0:
                        severity = VulnerabilitySeverity.HIGH
                    elif cvss_score >= 4.0:
                        severity = VulnerabilitySeverity.MEDIUM
                    else:
                        severity = VulnerabilitySeverity.LOW
            
            # Parse dates
            published_date = datetime.utcnow()
            if "published" in vuln_data:
                try:
                    published_date = datetime.fromisoformat(vuln_data["published"].replace("Z", "+00:00"))
                except:
                    pass
            
            # Extract affected versions and fix
            affected_versions = []
            fixed_version = None
            
            for affected in vuln_data.get("affected", []):
                for range_data in affected.get("ranges", []):
                    for event in range_data.get("events", []):
                        if "introduced" in event:
                            affected_versions.append(event["introduced"])
                        elif "fixed" in event:
                            fixed_version = event["fixed"]
            
            return Vulnerability(
                cve_id=cve_id,
                severity=severity,
                cvss_score=cvss_score,
                description=summary,
                affected_versions=affected_versions,
                fixed_version=fixed_version,
                published_date=published_date,
                references=[ref.get("url", "") for ref in vuln_data.get("references", [])]
            )
            
        except Exception as e:
            logger.error(f"Error parsing OSV vulnerability: {e}")
            return None
    
    def _parse_nvd_vulnerability(self, vuln_data: Dict) -> Optional[Vulnerability]:
        """Parse NVD vulnerability data"""
        try:
            cve_data = vuln_data.get("cve", {})
            cve_id = cve_data.get("id", "")
            
            # Extract description
            descriptions = cve_data.get("descriptions", [])
            description = ""
            for desc in descriptions:
                if desc.get("lang") == "en":
                    description = desc.get("value", "")
                    break
            
            # Extract CVSS score
            metrics = cve_data.get("metrics", {})
            cvss_score = 5.0
            severity = VulnerabilitySeverity.MEDIUM
            
            for metric_type in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
                if metric_type in metrics:
                    metric_data = metrics[metric_type][0]  # Take first metric
                    cvss_data = metric_data.get("cvssData", {})
                    cvss_score = float(cvss_data.get("baseScore", 5.0))
                    
                    if cvss_score >= 9.0:
                        severity = VulnerabilitySeverity.CRITICAL
                    elif cvss_score >= 7.0:
                        severity = VulnerabilitySeverity.HIGH
                    elif cvss_score >= 4.0:
                        severity = VulnerabilitySeverity.MEDIUM
                    else:
                        severity = VulnerabilitySeverity.LOW
                    break
            
            # Parse publication date
            published_date = datetime.utcnow()
            if "published" in cve_data:
                try:
                    published_date = datetime.fromisoformat(cve_data["published"].replace("Z", "+00:00"))
                except:
                    pass
            
            # Extract references
            references = []
            for ref in cve_data.get("references", []):
                references.append(ref.get("url", ""))
            
            return Vulnerability(
                cve_id=cve_id,
                severity=severity,
                cvss_score=cvss_score,
                description=description,
                affected_versions=[],  # NVD doesn't provide this easily
                fixed_version=None,
                published_date=published_date,
                references=references
            )
            
        except Exception as e:
            logger.error(f"Error parsing NVD vulnerability: {e}")
            return None

class ComponentDiscovery:
    """Discover components from various package managers"""
    
    async def discover_python_components(self, project_path: Path) -> List[Component]:
        """Discover Python components"""
        components = []
        
        # Check requirements.txt
        req_file = project_path / "requirements.txt"
        if req_file.exists():
            with open(req_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        component = self._parse_python_requirement(line)
                        if component:
                            components.append(component)
        
        # Check pyproject.toml
        pyproject_file = project_path / "pyproject.toml"
        if pyproject_file.exists():
            try:
                data = toml.load(pyproject_file)
                deps = data.get("project", {}).get("dependencies", [])
                for dep in deps:
                    component = self._parse_python_requirement(dep)
                    if component:
                        components.append(component)
            except Exception as e:
                logger.error(f"Error parsing pyproject.toml: {e}")
        
        # Get installed package info
        try:
            result = subprocess.run(
                ["pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                installed_packages = json.loads(result.stdout)
                installed_dict = {pkg["name"].lower(): pkg["version"] for pkg in installed_packages}
                
                # Enhance components with installed versions
                for component in components:
                    if component.name.lower() in installed_dict:
                        component.version = installed_dict[component.name.lower()]
        
        except Exception as e:
            logger.error(f"Error getting installed Python packages: {e}")
        
        return components
    
    async def discover_javascript_components(self, project_path: Path) -> List[Component]:
        """Discover JavaScript/Node.js components"""
        components = []
        
        package_json = project_path / "package.json"
        if package_json.exists():
            try:
                with open(package_json) as f:
                    data = json.load(f)
                
                # Regular dependencies
                for name, version in data.get("dependencies", {}).items():
                    component = Component(
                        name=name,
                        version=self._clean_version(version),
                        component_type=ComponentType.LIBRARY,
                        supplier="npm",
                        purl=f"pkg:npm/{name}@{self._clean_version(version)}"
                    )
                    components.append(component)
                
                # Dev dependencies
                for name, version in data.get("devDependencies", {}).items():
                    component = Component(
                        name=f"{name} (dev)",
                        version=self._clean_version(version),
                        component_type=ComponentType.LIBRARY,
                        supplier="npm",
                        purl=f"pkg:npm/{name}@{self._clean_version(version)}"
                    )
                    components.append(component)
            
            except Exception as e:
                logger.error(f"Error parsing package.json: {e}")
        
        return components
    
    async def discover_rust_components(self, project_path: Path) -> List[Component]:
        """Discover Rust components"""
        components = []
        
        cargo_toml = project_path / "Cargo.toml"
        if cargo_toml.exists():
            try:
                data = toml.load(cargo_toml)
                
                # Dependencies
                for name, version_data in data.get("dependencies", {}).items():
                    if isinstance(version_data, str):
                        version = version_data
                    elif isinstance(version_data, dict):
                        version = version_data.get("version", "unknown")
                    else:
                        continue
                    
                    component = Component(
                        name=name,
                        version=self._clean_version(version),
                        component_type=ComponentType.LIBRARY,
                        supplier="crates.io",
                        purl=f"pkg:cargo/{name}@{self._clean_version(version)}"
                    )
                    components.append(component)
            
            except Exception as e:
                logger.error(f"Error parsing Cargo.toml: {e}")
        
        return components
    
    def _parse_python_requirement(self, requirement: str) -> Optional[Component]:
        """Parse Python requirement string"""
        try:
            # Simple parsing - in production, use packaging library
            requirement = requirement.strip()
            
            # Handle version specifiers
            for op in [">=", "<=", "==", ">", "<", "~="]:
                if op in requirement:
                    name, version = requirement.split(op, 1)
                    name = name.strip()
                    version = version.strip()
                    
                    return Component(
                        name=name,
                        version=self._clean_version(version),
                        component_type=ComponentType.LIBRARY,
                        supplier="PyPI",
                        purl=f"pkg:pypi/{name}@{self._clean_version(version)}"
                    )
            
            # No version specified
            return Component(
                name=requirement,
                version="latest",
                component_type=ComponentType.LIBRARY,
                supplier="PyPI",
                purl=f"pkg:pypi/{requirement}@latest"
            )
            
        except Exception as e:
            logger.error(f"Error parsing requirement '{requirement}': {e}")
            return None
    
    def _clean_version(self, version: str) -> str:
        """Clean version string"""
        # Remove common prefixes
        version = version.strip()
        for prefix in ["^", "~", ">=", "<=", "==", ">", "<"]:
            if version.startswith(prefix):
                version = version[len(prefix):].strip()
        
        return version or "unknown"

class ArtifactSigner:
    """Sign and verify build artifacts"""
    
    def __init__(self, private_key_path: Optional[str] = None):
        self.private_key_path = private_key_path
        self._private_key = None
        self._public_key = None
        
        if private_key_path and Path(private_key_path).exists():
            self._load_keys()
        else:
            self._generate_keys()
    
    def _generate_keys(self):
        """Generate RSA key pair"""
        logger.info("Generating new RSA key pair for artifact signing")
        
        self._private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        self._public_key = self._private_key.public_key()
        
        # Save keys
        keys_dir = Path("keys")
        keys_dir.mkdir(exist_ok=True)
        
        # Private key
        private_pem = self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        with open(keys_dir / "signing_private.pem", "wb") as f:
            f.write(private_pem)
        
        # Public key
        public_pem = self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        with open(keys_dir / "signing_public.pem", "wb") as f:
            f.write(public_pem)
        
        logger.info("Keys saved to keys/ directory")
    
    def _load_keys(self):
        """Load existing keys"""
        try:
            with open(self.private_key_path, "rb") as f:
                self._private_key = load_pem_private_key(f.read(), password=None)
                self._public_key = self._private_key.public_key()
            
            logger.info(f"Loaded signing keys from {self.private_key_path}")
        except Exception as e:
            logger.error(f"Error loading keys: {e}")
            raise
    
    def sign_data(self, data: bytes) -> bytes:
        """Sign data with private key"""
        if not self._private_key:
            raise ValueError("Private key not available")
        
        signature = self._private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return signature
    
    def verify_signature(self, data: bytes, signature: bytes, public_key_pem: bytes = None) -> bool:
        """Verify signature"""
        try:
            if public_key_pem:
                public_key = load_pem_public_key(public_key_pem)
            else:
                public_key = self._public_key
            
            public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except Exception:
            return False
    
    def get_public_key_pem(self) -> bytes:
        """Get public key in PEM format"""
        if not self._public_key:
            raise ValueError("Public key not available")
        
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    def sign_file(self, file_path: Path, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Sign a file and create signature manifest"""
        
        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path)
        
        # Create manifest
        manifest = {
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "hash_algorithm": "sha256",
            "file_hash": file_hash,
            "metadata": metadata or {},
            "signed_at": datetime.utcnow().isoformat(),
            "signer": "aivillage-security-system",
            "public_key_fingerprint": self._get_public_key_fingerprint()
        }
        
        # Sign the manifest
        manifest_json = json.dumps(manifest, sort_keys=True)
        signature = self.sign_data(manifest_json.encode())
        
        signed_manifest = {
            **manifest,
            "signature": signature.hex(),
            "public_key": self.get_public_key_pem().decode()
        }
        
        # Write signature file
        signature_path = file_path.with_suffix(file_path.suffix + ".sig")
        with open(signature_path, "w") as f:
            json.dump(signed_manifest, f, indent=2)
        
        logger.info(f"Signed {file_path} -> {signature_path}")
        return signed_manifest
    
    def verify_file(self, file_path: Path) -> Dict[str, Any]:
        """Verify file signature and integrity"""
        signature_path = file_path.with_suffix(file_path.suffix + ".sig")
        
        if not signature_path.exists():
            return {"verified": False, "error": "No signature file found"}
        
        try:
            # Load signature manifest
            with open(signature_path) as f:
                signed_manifest = json.load(f)
            
            # Verify file hash
            current_hash = self._calculate_file_hash(file_path)
            if current_hash != signed_manifest["file_hash"]:
                return {"verified": False, "error": "File hash mismatch"}
            
            # Verify signature
            manifest_without_sig = {k: v for k, v in signed_manifest.items() 
                                  if k not in ["signature", "public_key"]}
            manifest_json = json.dumps(manifest_without_sig, sort_keys=True)
            
            signature = bytes.fromhex(signed_manifest["signature"])
            public_key_pem = signed_manifest["public_key"].encode()
            
            signature_valid = self.verify_signature(
                manifest_json.encode(),
                signature,
                public_key_pem
            )
            
            if not signature_valid:
                return {"verified": False, "error": "Invalid signature"}
            
            return {
                "verified": True,
                "signer": signed_manifest["signer"],
                "signed_at": signed_manifest["signed_at"],
                "metadata": signed_manifest["metadata"],
                "public_key_fingerprint": signed_manifest["public_key_fingerprint"]
            }
            
        except Exception as e:
            return {"verified": False, "error": f"Verification failed: {e}"}
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        hash_obj = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    def _get_public_key_fingerprint(self) -> str:
        """Get public key fingerprint"""
        public_key_pem = self.get_public_key_pem()
        fingerprint = hashlib.sha256(public_key_pem).hexdigest()[:16]
        return fingerprint

class EnhancedSBOMGenerator:
    """Enhanced SBOM generator with security analysis"""
    
    def __init__(self, vulnerability_scanner: VulnerabilityScanner = None):
        self.scanner = vulnerability_scanner or VulnerabilityScanner()
        self.discovery = ComponentDiscovery()
        self.signer = ArtifactSigner()
        
    async def generate_comprehensive_sbom(self, 
                                         project_path: Path,
                                         include_vulnerabilities: bool = True,
                                         include_licenses: bool = True) -> Dict[str, Any]:
        """Generate comprehensive SBOM with security analysis"""
        
        logger.info(f"Generating comprehensive SBOM for {project_path}")
        
        # Discover components
        all_components = []
        
        # Python components
        python_components = await self.discovery.discover_python_components(project_path)
        all_components.extend(python_components)
        
        # JavaScript components
        js_components = await self.discovery.discover_javascript_components(project_path)
        all_components.extend(js_components)
        
        # Rust components
        rust_components = await self.discovery.discover_rust_components(project_path)
        all_components.extend(rust_components)
        
        logger.info(f"Discovered {len(all_components)} components")
        
        # Enhance components with security data
        if include_vulnerabilities:
            logger.info("Scanning for vulnerabilities...")
            await self._scan_vulnerabilities(all_components)
        
        if include_licenses:
            logger.info("Analyzing license compliance...")
            self._analyze_licenses(all_components)
        
        # Calculate supply chain risk
        logger.info("Calculating supply chain risk...")
        for component in all_components:
            component.calculate_supply_chain_risk()
        
        # Generate SBOM
        sbom = self._create_cyclone_dx_sbom(project_path, all_components)
        
        # Add security analysis
        sbom["security_analysis"] = self._create_security_analysis(all_components)
        
        logger.info("SBOM generation completed")
        return sbom
    
    async def _scan_vulnerabilities(self, components: List[Component]):
        """Scan components for vulnerabilities"""
        for component in components:
            try:
                vulnerabilities = await self.scanner.scan_component(component)
                for vuln in vulnerabilities:
                    component.add_vulnerability(vuln)
                
                if vulnerabilities:
                    logger.info(f"Found {len(vulnerabilities)} vulnerabilities in {component.name}")
                    
            except Exception as e:
                logger.error(f"Error scanning {component.name}: {e}")
    
    def _analyze_licenses(self, components: List[Component]):
        """Analyze license compliance for components"""
        for component in components:
            # This would integrate with license databases in production
            # For now, we'll do basic license detection
            
            if not component.licenses:
                # Try to detect license from common patterns
                common_licenses = {
                    'mit': ['MIT'],
                    'apache': ['Apache-2.0'],
                    'bsd': ['BSD-3-Clause'],
                    'gpl': ['GPL-3.0'],
                    'lgpl': ['LGPL-3.0']
                }
                
                name_lower = component.name.lower()
                for pattern, licenses in common_licenses.items():
                    if pattern in name_lower:
                        component.licenses = licenses
                        break
            
            component.assess_license_risk()
    
    def _create_cyclone_dx_sbom(self, project_path: Path, components: List[Component]) -> Dict[str, Any]:
        """Create CycloneDX format SBOM"""
        
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "serialNumber": f"urn:uuid:{self._generate_uuid()}",
            "version": 1,
            "metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "tools": [
                    {
                        "vendor": "AIVillage",
                        "name": "enhanced-sbom-generator", 
                        "version": "1.0.0"
                    }
                ],
                "component": {
                    "type": "application",
                    "bom-ref": "aivillage-app",
                    "name": project_path.name,
                    "version": "1.0.0",
                    "description": "AIVillage distributed AI platform"
                }
            },
            "components": []
        }
        
        # Add components
        for component in components:
            component_data = {
                "type": component.component_type.value,
                "bom-ref": component.bom_ref,
                "name": component.name,
                "version": component.version
            }
            
            if component.supplier:
                component_data["supplier"] = {"name": component.supplier}
            
            if component.description:
                component_data["description"] = component.description
            
            if component.licenses:
                component_data["licenses"] = [
                    {"license": {"name": license_name}} 
                    for license_name in component.licenses
                ]
            
            if component.cpe:
                component_data["cpe"] = component.cpe
            
            if component.purl:
                component_data["purl"] = component.purl
            
            if component.hashes:
                component_data["hashes"] = [
                    {"alg": alg, "content": value}
                    for alg, value in component.hashes.items()
                ]
            
            if component.external_references:
                component_data["externalReferences"] = component.external_references
            
            sbom["components"].append(component_data)
        
        return sbom
    
    def _create_security_analysis(self, components: List[Component]) -> Dict[str, Any]:
        """Create security analysis summary"""
        
        total_vulns = sum(len(c.vulnerabilities) for c in components)
        critical_vulns = sum(
            len([v for v in c.vulnerabilities if v.severity == VulnerabilitySeverity.CRITICAL])
            for c in components
        )
        high_vulns = sum(
            len([v for v in c.vulnerabilities if v.severity == VulnerabilitySeverity.HIGH])
            for c in components
        )
        
        license_risk_counts = {
            risk.value: len([c for c in components if c.license_risk == risk])
            for risk in LicenseRisk
        }
        
        high_risk_components = [
            {
                "name": c.name,
                "version": c.version,
                "supply_chain_risk": round(c.supply_chain_risk, 2),
                "vulnerabilities": len(c.vulnerabilities),
                "license_risk": c.license_risk.value
            }
            for c in components
            if c.supply_chain_risk > 7.0 or len(c.vulnerabilities) > 0
        ]
        
        return {
            "total_components": len(components),
            "vulnerability_summary": {
                "total_vulnerabilities": total_vulns,
                "critical_vulnerabilities": critical_vulns,
                "high_vulnerabilities": high_vulns,
                "components_with_vulns": len([c for c in components if c.vulnerabilities])
            },
            "license_risk_summary": license_risk_counts,
            "supply_chain_risk": {
                "average_risk": round(
                    sum(c.supply_chain_risk for c in components) / len(components) if components else 0,
                    2
                ),
                "high_risk_components": len([c for c in components if c.supply_chain_risk > 7.0]),
                "components_needing_updates": len([
                    c for c in components 
                    if c.vulnerabilities and any(v.fixed_version for v in c.vulnerabilities)
                ])
            },
            "high_risk_components": high_risk_components[:10],  # Top 10
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    def _generate_uuid(self) -> str:
        """Generate UUID for SBOM"""
        import uuid
        return str(uuid.uuid4())
    
    async def save_sbom(self, sbom: Dict[str, Any], output_path: Path, sign: bool = True) -> Path:
        """Save SBOM to file and optionally sign it"""
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write SBOM file
        with open(output_path, "w") as f:
            json.dump(sbom, f, indent=2, default=str)
        
        logger.info(f"SBOM saved to {output_path}")
        
        # Sign the file
        if sign:
            signature_data = self.signer.sign_file(
                output_path,
                metadata={
                    "sbom_format": "CycloneDX",
                    "total_components": sbom.get("security_analysis", {}).get("total_components", 0),
                    "vulnerability_count": sbom.get("security_analysis", {}).get("vulnerability_summary", {}).get("total_vulnerabilities", 0)
                }
            )
            logger.info(f"SBOM signed with fingerprint: {signature_data['public_key_fingerprint']}")
        
        return output_path

# CLI interface and example usage
async def main():
    """Example usage of enhanced SBOM generator"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate enhanced SBOM")
    parser.add_argument("project_path", type=Path, help="Path to project")
    parser.add_argument("--output", type=Path, default=Path("sbom/aivillage-sbom.json"), help="Output file")
    parser.add_argument("--no-vulns", action="store_true", help="Skip vulnerability scanning")
    parser.add_argument("--no-licenses", action="store_true", help="Skip license analysis")
    parser.add_argument("--no-sign", action="store_true", help="Skip signing")
    
    args = parser.parse_args()
    
    # Initialize generator
    scanner = VulnerabilityScanner()
    generator = EnhancedSBOMGenerator(scanner)
    
    try:
        # Generate SBOM
        sbom = await generator.generate_comprehensive_sbom(
            args.project_path,
            include_vulnerabilities=not args.no_vulns,
            include_licenses=not args.no_licenses
        )
        
        # Save SBOM
        output_path = await generator.save_sbom(
            sbom,
            args.output,
            sign=not args.no_sign
        )
        
        # Print summary
        security_analysis = sbom.get("security_analysis", {})
        print(f"\n📊 SBOM Generation Summary:")
        print(f"   Components: {security_analysis.get('total_components', 0)}")
        print(f"   Vulnerabilities: {security_analysis.get('vulnerability_summary', {}).get('total_vulnerabilities', 0)}")
        print(f"   Critical: {security_analysis.get('vulnerability_summary', {}).get('critical_vulnerabilities', 0)}")
        print(f"   High: {security_analysis.get('vulnerability_summary', {}).get('high_vulnerabilities', 0)}")
        print(f"   High Risk Components: {security_analysis.get('supply_chain_risk', {}).get('high_risk_components', 0)}")
        print(f"   Output: {output_path}")
        
        if not args.no_sign:
            print(f"   Signature: {output_path.with_suffix(output_path.suffix + '.sig')}")
        
    except Exception as e:
        logger.error(f"SBOM generation failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())